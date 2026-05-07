import numpy as np
import torch
import os
import h5py
import yaml
import argparse
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        return [f.strip() for f in f.readlines()]

def load_system_matrix(h5_path: str, device: torch.device) -> torch.Tensor:
    """Auto-detects sparse/dense HDF5 and returns a PyTorch tensor."""
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            # Load Sparse CSR format
            indptr = torch.tensor(h5f["indptr"][:], dtype=torch.int32, device=device)
            indices = torch.tensor(h5f["indices"][:], dtype=torch.int32, device=device)
            data = torch.tensor(h5f["data"][:], dtype=torch.float32, device=device)
            shape = tuple(h5f.attrs["shape"])
            
            return torch.sparse_csr_tensor(indptr, indices, data, size=shape, device=device)
        elif "ppdfs" in h5f:
            # Load Legacy Dense format
            return torch.tensor(h5f["ppdfs"][:], dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown matrix format in {h5_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flist = get_flist(cfg['paths']['flist_path'])
    
    img_size = cfg['geometry']['img_dim']
    sfov_expected = img_size * img_size

    # --- Phantom Loading ---
    phantom_data = torch.load(cfg['paths']['phantom_path'], map_location=device, weights_only=False)
    phantom_tensor = phantom_data["Phantom tensor"]
    
    h, w = phantom_tensor.shape
    pad_h, pad_w = (img_size - h) // 2, (img_size - w) // 2
    phantom_padded = torch.nn.functional.pad(phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
    
    phantom_flat = phantom_padded.view(-1, 1).to(device)

    all_projs = []
    progress = Progress(TextColumn("[blue]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())

    with torch.no_grad(), progress:
        task = progress.add_task("Forward Projecting...", total=len(flist))
        for fname in flist:
            matrix_chunk = load_system_matrix(fname, device)
            
            # --- OPTIMIZATION: Explicit Sparse/Dense Dispatch ---
            if matrix_chunk.is_sparse_csr:
                proj_chunk = torch.sparse.mm(matrix_chunk, phantom_flat)
            else:
                proj_chunk = torch.matmul(matrix_chunk, phantom_flat)
            
            # Squeeze back to 1D and save to CPU RAM to prevent VRAM overflow
            all_projs.append(proj_chunk.squeeze().cpu())
            progress.update(task, advance=1)

    # Stack all layouts
    final_projs = torch.stack(all_projs, dim=0)
    np.save(cfg['paths']['projs_path'], final_projs.numpy())
    print(f"Projections saved to: {cfg['paths']['projs_path']}")

if __name__ == "__main__":
    main()