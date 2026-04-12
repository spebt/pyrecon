"""
MAP-TV Reconstruction for SPECT (Compressive Sensing Formulation).

Implements the EM-TV operator splitting algorithm:
  - Outer loop: ML-EM update for Poisson likelihood (identical to MLEM).
  - Inner loop: Chambolle-Pock primal-dual TV proximal denoising step.

Objective:
    Phi(x) = Poisson_NLL(Ax + r, y) + beta * TV(x),  subject to x >= 0.

Reference: new_recon.md — MAP-TV Reconstruction for SPECT specification.
"""

import numpy as np
import time
import torch
import os
import h5py
import yaml
import argparse
import resource
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

from tv_ops import tv_proximal_chambolle_pock


# ---------------------------------------------------------------------------
# I/O helpers (identical to recon_mlem for compatibility)
# ---------------------------------------------------------------------------

def load_system_matrix(h5_path: str, device: torch.device) -> torch.Tensor:
    """Auto-detects sparse/dense HDF5 and returns a PyTorch tensor."""
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            indptr  = torch.tensor(h5f["indptr"][:],  dtype=torch.int32,   device=device)
            indices = torch.tensor(h5f["indices"][:], dtype=torch.int32,   device=device)
            data    = torch.tensor(h5f["data"][:],    dtype=torch.float32, device=device)
            shape   = tuple(h5f.attrs["shape"])
            return torch.sparse_csr_tensor(indptr, indices, data, size=shape, device=device)
        elif "ppdfs" in h5f:
            return torch.tensor(h5f["ppdfs"][:], dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown matrix format in {h5_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAP-TV SPECT Reconstruction")
    parser.add_argument("--config", default="configs/base_config.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # --- 1. Setup ---
    img_dim = cfg["geometry"]["img_dim"]
    sfov    = img_dim * img_dim

    n_outer  = int(cfg["map_tv"]["n_outer"])
    n_inner  = int(cfg["map_tv"]["n_inner"])
    beta     = float(cfg["map_tv"]["beta"])          # global TV weight
    tau      = float(cfg["map_tv"]["tau"])            # primal step size
    sigma    = float(cfg["map_tv"]["sigma"])          # dual step size
    theta    = float(cfg["map_tv"]["theta"])          # extrapolation factor
    eps      = float(cfg["map_tv"]["eps"])            # numerical floor
    save_every      = int(cfg["map_tv"]["save_every"])
    convergence_tol = float(cfg["map_tv"]["convergence_tol"])
    cache_data      = bool(cfg["map_tv"]["cache_data"])

    # Stability check: τσ < 1/8 for 2D TV (6 neighbors → 1/12 for 3D)
    if tau * sigma >= 1.0 / 8.0:
        raise ValueError(
            f"Chambolle-Pock stability violated: τσ={tau*sigma:.4f} must be < 1/8=0.125. "
            f"Reduce tau or sigma."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Image: {img_dim}x{img_dim}  |  β={beta}  |  τ={tau}  |  σ={sigma}")

    flist      = [f.strip() for f in open(cfg["paths"]["flist_path"], "r")]
    pdata_full = torch.from_numpy(np.load(cfg["paths"]["projs_path"])).to(device)

    # Scatter/randoms background (set to zero if not provided)
    r_background = float(cfg["map_tv"].get("background", 0.0))

    # --- 2. Pre-compute sensitivity map & optionally cache matrices ---
    print("Pre-calculating Sensitivity Map...")
    estimate        = torch.ones((sfov, 1), device=device, dtype=torch.float32)
    sensitivity_map = torch.zeros((sfov, 1), device=device, dtype=torch.float32)
    back_proj       = torch.zeros((sfov, 1), device=device, dtype=torch.float32)
    cached_matrices = []

    with torch.no_grad():
        for fname in flist:
            m_chunk = load_system_matrix(fname, device)
            if m_chunk.is_sparse_csr:
                m_chunk_t   = m_chunk.t().to_sparse_csr()
                sproj       = m_chunk.shape[0]
                ones_vec    = torch.ones((sproj, 1), device=device, dtype=torch.float32)
                sensitivity_map += torch.sparse.mm(m_chunk_t, ones_vec)
            else:
                m_chunk_t   = m_chunk.t()
                sproj       = m_chunk.shape[0]
                ones_vec    = torch.ones((sproj, 1), device=device, dtype=torch.float32)
                sensitivity_map += torch.matmul(m_chunk_t, ones_vec)

            if cache_data:
                cached_matrices.append((m_chunk, m_chunk_t))

    # Avoid division by zero in EM update
    sensitivity_map.clamp_(min=eps)

    estimates_history = []
    progress = Progress(
        TextColumn("[green]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    # --- 3. Main MAP-TV Loop ---
    with progress, torch.no_grad():
        task = progress.add_task("MAP-TV Reconstructing...", total=n_outer)

        for it in range(n_outer):
            prev_est = estimate.clone()
            back_proj.zero_()

            # ----------------------------------------------------------------
            # Step 1: ML-EM likelihood update  (Poisson forward model)
            # ----------------------------------------------------------------
            for i, fname in enumerate(flist):
                if cache_data:
                    m_chunk, m_chunk_t = cached_matrices[i]
                else:
                    m_chunk   = load_system_matrix(fname, device)
                    m_chunk_t = m_chunk.t().to_sparse_csr() if m_chunk.is_sparse_csr else m_chunk.t()

                # Forward project: λ = A x + r
                if m_chunk.is_sparse_csr:
                    lambda_chunk = torch.sparse.mm(m_chunk, estimate) + r_background
                else:
                    lambda_chunk = torch.matmul(m_chunk, estimate) + r_background

                lambda_chunk.clamp_(min=eps)

                # Ratio: y / λ
                ratio_chunk = pdata_full[i].view(-1, 1) / lambda_chunk

                # Back-project ratio: A^T (y/λ)
                if m_chunk_t.is_sparse_csr:
                    back_proj += torch.sparse.mm(m_chunk_t, ratio_chunk)
                else:
                    back_proj += torch.matmul(m_chunk_t, ratio_chunk)

            # x_em = x * (A^T(y/λ)) / (A^T 1)
            x_em_flat = estimate * back_proj / sensitivity_map   # (sfov, 1)

            # ----------------------------------------------------------------
            # Step 2: TV Proximal Step (Chambolle-Pock inner loop)
            # Solves: x^{k+1} = argmin_x 0.5||x - x_em||^2 + alpha*TV(x)
            # ----------------------------------------------------------------
            # Reshape to (H, W) for spatial TV operators
            x_em_2d = x_em_flat.view(img_dim, img_dim)

            x_tv_2d = tv_proximal_chambolle_pock(
                x_em   = x_em_2d,
                alpha  = beta,       # alpha in the inner problem = beta (global weight)
                tau    = tau,
                sigma  = sigma,
                theta  = theta,
                n_inner= n_inner,
                eps    = eps,
            )

            # Flatten back to column vector
            estimate = x_tv_2d.view(sfov, 1)

            # ----------------------------------------------------------------
            # Bookkeeping
            # ----------------------------------------------------------------
            if it % save_every == 0:
                estimates_history.append(estimate.view(img_dim, img_dim).cpu().numpy())

            diff = torch.norm(estimate - prev_est) / torch.norm(prev_est).clamp(min=eps)
            if diff < convergence_tol:
                print(f"\nConvergence reached at outer iteration {it}  (diff={diff:.2e})")
                break

            progress.update(
                task, advance=1,
                description=f"MAP-TV [outer={it+1}/{n_outer}  diff={diff:.2e}]"
            )

    # --- 4. Save ---
    np.savez_compressed(cfg["paths"]["recon_out_path"], estimates=np.array(estimates_history))
    print(f"Reconstruction saved to: {cfg['paths']['recon_out_path']}")

    # --- 5. Memory Report ---
    print("-" * 40)
    print("RUNTIME MEMORY REPORT:")
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_gb = max_rss_kb / (1024 ** 2)
    print(f"  -> Peak CPU RAM: ~{max_rss_gb:.2f} GB")
    if device.type == "cuda":
        max_vram_bytes = torch.cuda.max_memory_allocated(device)
        max_vram_gb    = max_vram_bytes / (1024 ** 3)
        print(f"  -> Peak GPU VRAM: ~{max_vram_gb:.2f} GB")
    print("-" * 40)


if __name__ == "__main__":
    main()
