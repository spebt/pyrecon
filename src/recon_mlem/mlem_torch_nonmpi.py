import numpy as np
import time
import torch
import os
import h5py
import yaml
import argparse
import resource
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def load_system_matrix(h5_path: str, device: torch.device) -> torch.Tensor:
    """Auto-detects sparse/dense HDF5 and returns a PyTorch tensor."""
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            indptr = torch.tensor(h5f["indptr"][:], dtype=torch.int32, device=device)
            indices = torch.tensor(h5f["indices"][:], dtype=torch.int32, device=device)
            data = torch.tensor(h5f["data"][:], dtype=torch.float32, device=device)
            shape = tuple(h5f.attrs["shape"])
            return torch.sparse_csr_tensor(indptr, indices, data, size=shape, device=device)
        elif "ppdfs" in h5f:
            return torch.tensor(h5f["ppdfs"][:], dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown matrix format in {h5_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- 1. Geometry & Hardware Setup ---
    img_dim = cfg['geometry']['img_dim']
    sfov = img_dim * img_dim
    
    n_iterations = cfg['mlem']['iterations']
    cache_data = cfg['mlem']['cache_data']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flist = [f.strip() for f in open(cfg['paths']['flist_path'], "r")]
    pdata_full = torch.from_numpy(np.load(cfg['paths']['projs_path'])).to(device)

    estimate = torch.ones((sfov, 1), device=device, dtype=torch.float32)
    sensitivity_map = torch.zeros((sfov, 1), device=device, dtype=torch.float32)
    
    # Pre-allocate back_proj to avoid reallocating memory every iteration
    back_proj = torch.zeros((sfov, 1), device=device, dtype=torch.float32)
    
    cached_matrices = [] 

    # --- 2. Pre-calculation & Caching ---
    print(f"System: {img_dim}x{img_dim} FOV")
    print("Pre-calculating Sensitivity Map and Caching Matrices...")
    
    with torch.no_grad():
        for i, fname in enumerate(flist):
            m_chunk = load_system_matrix(fname, device)
            
            # --- OPTIMIZATION: The CSR Transpose Trick ---
            # .t() on CSR creates CSC. Converting it back to CSR explicitly guarantees
            # the fastest possible cuSPARSE execution during the Back Projection loop.
            if m_chunk.is_sparse_csr:
                m_chunk_t = m_chunk.t().to_sparse_csr()
                # Sensitivity map: A^T * ones
                sproj = m_chunk.shape[0]
                ones_vec = torch.ones((sproj, 1), device=device, dtype=torch.float32)
                sensitivity_map += torch.sparse.mm(m_chunk_t, ones_vec)
            else:
                m_chunk_t = m_chunk.t()
                sproj = m_chunk.shape[0]
                ones_vec = torch.ones((sproj, 1), device=device, dtype=torch.float32)
                sensitivity_map += torch.matmul(m_chunk_t, ones_vec)
                
            if cache_data: 
                cached_matrices.append((m_chunk, m_chunk_t))

    # Prevent division by zero
    sensitivity_map[sensitivity_map == 0] = 1.0
    estimates_history = []
    
    progress = Progress(TextColumn("[green]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())

    # --- 3. Main MLEM Loop ---
    with progress, torch.no_grad():
        task = progress.add_task("MLEM Reconstructing...", total=n_iterations)
        for it in range(n_iterations):
            prev_est = estimate.clone()
            
            # --- OPTIMIZATION: In-place zeroing instead of re-allocation ---
            back_proj.zero_()

            for i, fname in enumerate(flist):
                # Retrieve matrices
                if cache_data:
                    m_chunk, m_chunk_t = cached_matrices[i]
                else:
                    m_chunk = load_system_matrix(fname, device)
                    m_chunk_t = m_chunk.t().to_sparse_csr() if m_chunk.is_sparse_csr else m_chunk.t()
                
                # Step 1: Forward Project (y = A * x)
                if m_chunk.is_sparse_csr:
                    y_chunk = torch.sparse.mm(m_chunk, estimate)
                else:
                    y_chunk = torch.matmul(m_chunk, estimate)
                    
                y_chunk[y_chunk == 0] = 1e-9 
                
                # Step 2: Ratio (p / y)
                r_chunk = pdata_full[i].view(-1, 1) / y_chunk 
                
                # Step 3: Back-project (BP = A^T * r)
                # Using the optimized CSR-format transposed matrix
                if m_chunk_t.is_sparse_csr:
                    bp_chunk = torch.sparse.mm(m_chunk_t, r_chunk)
                else:
                    bp_chunk = torch.matmul(m_chunk_t, r_chunk)
                    
                back_proj += bp_chunk

            # --- OPTIMIZATION: In-place multiplicative update ---
            # Replaces: estimate = estimate * (back_proj / sensitivity_map)
            estimate.mul_(back_proj).div_(sensitivity_map)
            
            # Record progress
            if it % cfg['mlem']['save_every'] == 0:
                estimates_history.append(estimate.view(img_dim, img_dim).cpu().numpy())

            diff = torch.norm(estimate - prev_est) / torch.norm(prev_est)
            if diff < float(cfg['mlem']['convergence_tol']):
                print(f"Convergence reached at iteration {it}")
                break
            
            progress.update(task, advance=1, description=f"MLEM (diff: {diff:.2e})")

    # --- 4. Final Save ---
    np.savez_compressed(cfg['paths']['recon_out_path'], estimates=np.array(estimates_history))
    print(f"Reconstruction saved to: {cfg['paths']['recon_out_path']}")

    # Print Peak Memory Usage
    print("-" * 40)
    print("RUNTIME MEMORY REPORT:")
    
    # resource.getrusage returns peak memory in Kilobytes on Linux (like your HPC cluster)
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_gb = max_rss_kb / (1024 ** 2) 
    print(f"  -> Peak CPU RAM: ~{max_rss_gb:.2f} GB")
    
    if device.type == "cuda":
        max_vram_bytes = torch.cuda.max_memory_allocated(device)
        max_vram_gb = max_vram_bytes / (1024 ** 3)
        print(f"  -> Peak GPU VRAM: ~{max_vram_gb:.2f} GB")
    print("-" * 40)

if __name__ == "__main__":
    main()