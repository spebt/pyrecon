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

def get_detector_count(layout_file: str) -> int:
    """Dynamically determines the number of detectors from the layout file."""
    if not os.path.exists(layout_file):
        raise FileNotFoundError(f"Layout file not found for detector counting: {layout_file}")
    
    # Load the .tensor file
    layout_data = torch.load(layout_file, map_location="cpu")
    
    # Grab the first available position to count detectors
    first_pos_key = list(layout_data["layouts"].keys())[0]
    detector_tensor = layout_data["layouts"][first_pos_key]["detector units"]
    
    # The first dimension of (N, Vertices, XY) is the number of detectors
    return detector_tensor.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flist = get_flist(cfg['paths']['flist_path'])
    
    # --- Dynamic Geometry Setup ---
    # We use the original .tensor file path from your paths config
    # You might need to add 'base_layout_path' to your config or use a relative path
    layout_path = cfg['paths'].get('base_layout_path', '../data/scanner_layouts/mph_hourglass_single_position_base_2mm_18pinholes.tensor')
    
    sproj = get_detector_count(layout_path)
    img_size = cfg['geometry']['img_dim']
    sfov_expected = img_size * img_size
    
    print(f"Detected {sproj} projection bins from layout.")

    # --- Phantom Loading ---
    phantom_data = torch.load(cfg['paths']['phantom_path'], map_location=device)
    phantom_tensor = phantom_data["Phantom tensor"]
    
    h, w = phantom_tensor.shape
    pad_h, pad_w = (img_size - h) // 2, (img_size - w) // 2
    phantom_padded = torch.nn.functional.pad(phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
    phantom_flat = phantom_padded.view(-1).to(device)

    all_projs = []
    progress = Progress(TextColumn("[blue]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())

    with torch.no_grad(), progress:
        task = progress.add_task("Forward Projecting...", total=len(flist))
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                matrix_chunk = torch.tensor(h5f["ppdfs"][:], device=device).view(1, sproj, sfov_expected)
                proj_chunk = torch.matmul(matrix_chunk, phantom_flat)
                all_projs.append(proj_chunk.cpu())
            progress.update(task, advance=1)

    final_projs = torch.cat(all_projs, dim=0)
    np.save(cfg['paths']['projs_path'], final_projs.numpy())
    print(f"Projections saved to: {cfg['paths']['projs_path']}")

if __name__ == "__main__":
    main()