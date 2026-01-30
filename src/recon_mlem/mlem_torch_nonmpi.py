import numpy as np
import time
import torch
import os
import h5py
import yaml
import argparse
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def get_detector_count(layout_file: str) -> int:
    """Dynamically determines the number of detectors from the layout file."""
    if not os.path.exists(layout_file):
        raise FileNotFoundError(f"Layout file missing: {layout_file}")
    
    layout_data = torch.load(layout_file, map_location="cpu", weights_only=False)
    first_pos_key = list(layout_data["layouts"].keys())[0]
    return layout_data["layouts"][first_pos_key]["detector units"].shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recon_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- 1. Dynamic Geometry & Hardware Setup ---
    img_dim = cfg['geometry']['img_dim']
    sfov = img_dim * img_dim
    
    # Auto-detect SPROJ from the base layout file
    layout_path = cfg['paths']['base_layout_path']
    sproj = get_detector_count(layout_path)
    
    n_iterations = cfg['mlem']['iterations']
    cache_data = cfg['mlem']['cache_data']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flist = [f.strip() for f in open(cfg['paths']['flist_path'], "r")]
    pdata_full = torch.from_numpy(np.load(cfg['paths']['projs_path'])).to(device)

    # Initialization
    estimate = torch.ones(sfov, device=device, dtype=torch.float32)
    sensitivity_map = torch.zeros(sfov, device=device, dtype=torch.float32)
    cached_matrices = [] 

    # --- 2. Pre-calculation & Caching ---
    print(f"System: {img_dim}x{img_dim} FOV | {sproj} Detectors")
    print("Pre-calculating Sensitivity Map...")
    
    with torch.no_grad():
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                # View matrix using the detected sproj
                data_t = torch.from_numpy(h5f["ppdfs"][:]).view(1, sproj, sfov)
                if cache_data: 
                    cached_matrices.append(data_t)
                sensitivity_map += torch.sum(data_t.to(device), dim=1).squeeze()

    sensitivity_map[sensitivity_map == 0] = 1.0
    estimates_history, times_history = [], []
    
    progress = Progress(TextColumn("[green]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())

    # --- 3. Main MLEM Loop ---
    with progress, torch.no_grad():
        task = progress.add_task("MLEM Reconstructing...", total=n_iterations)
        for it in range(n_iterations):
            it_start = time.time()
            prev_est = estimate.clone()
            back_proj = torch.zeros(sfov, device=device)

            for i, fname in enumerate(flist):
                # Retrieve matrix (from RAM or Disk)
                if cache_data:
                    m_chunk = cached_matrices[i].to(device)
                else:
                    m_chunk = torch.from_numpy(h5py.File(fname, "r")["ppdfs"][:]).view(1, sproj, sfov).to(device)
                
                # Step 1: Forward Project
                y_chunk = torch.matmul(m_chunk, estimate) 
                y_chunk[y_chunk == 0] = 1.0 
                
                # Step 2 & 3: Ratio and Back-project
                r_chunk = pdata_full[i].view(1, -1) / y_chunk 
                bp_chunk = torch.matmul(m_chunk.transpose(1, 2), r_chunk.unsqueeze(-1))
                back_proj += bp_chunk.squeeze()

            # Step 4: Multiplicative Update
            estimate = estimate * (back_proj / sensitivity_map)
            
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

if __name__ == "__main__":
    main()