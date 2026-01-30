import os
import torch
import h5py
import matplotlib.pyplot as plt
from torch import tensor, save as torch_save, float32 as torch_float32
from typing import Dict, List, Tuple

# Project imports based on your environment
from _disk_shape import fov_tensor_dict, single_disk

# ==============================================================================
# HELPERS (Derived from your provided logic)
# ==============================================================================

def mm_to_px_vec2(v_mm: torch.Tensor, mm_per_px_xy: torch.Tensor, n_pixels_xy: torch.Tensor) -> torch.Tensor:
    """Convert mm coordinates (centered at 0,0) to pixel indices."""
    return torch.round(v_mm / mm_per_px_xy + n_pixels_xy.to(v_mm.dtype) * 0.5).to(torch.long)

def safe_add_patch(img: torch.Tensor, patch: torch.Tensor, cx: int, cy: int) -> None:
    """Add a patch into an image centered at (cx, cy) with boundary clipping."""
    H, W = img.shape
    ph, pw = patch.shape
    x0, y0 = cx - ph // 2, cy - pw // 2
    x1, y1 = x0 + ph, y0 + pw
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(H, x1), min(W, y1)
    if sx0 >= sx1 or sy0 >= sy1: return
    px0, py0 = sx0 - x0, sy0 - y0
    px1, py1 = px0 + (sx1 - sx0), py0 + (sy1 - sy0)
    img[sx0:sx1, sy0:sy1] += patch[px0:px1, py0:py1]

# ==============================================================================
# MAIN GENERATION
# ==============================================================================

def generate_targeted_phantom():
    # 1. Configuration
    symnum = 5
    PHANTOM_POINTS = [(5.0, 5.0), (-5.0, -5.0)] # Point A source locations
    ROD_DIAMETER = 3.0 # mm
    PX_SIZE = 0.25 # mm/px
    GRID_SIZE = 280 # px (resulting in 70x70mm FOV)
    SS_FACTOR = 1 # Supersampling for smooth rods
    
    fov_size_mm = GRID_SIZE * PX_SIZE # 70.0 mm
    fov_n_pxs = (GRID_SIZE, GRID_SIZE)

    # Initialize FOV metadata
    fov_dict = fov_tensor_dict(fov_n_pxs, (fov_size_mm, fov_size_mm))
    
    # Create empty phantom (Zero background as requested)
    phantom = torch.zeros(fov_n_pxs, dtype=torch_float32)

    # 2. Generate Rod Patch
    # radius in pixels
    r_px = torch.tensor((ROD_DIAMETER * 0.5) / PX_SIZE, dtype=torch_float32)
    # Generate the supersampled disk
    rod_patch = single_disk(r_px, factor=SS_FACTOR)

    # 3. Place Rods
    centers_mm = tensor(PHANTOM_POINTS, dtype=torch_float32)
    centers_px = mm_to_px_vec2(centers_mm, tensor([PX_SIZE, PX_SIZE]), tensor(fov_n_pxs))
    
    for i in range(len(PHANTOM_POINTS)):
        cx, cy = int(centers_px[i, 0]), int(centers_px[i, 1])
        print(f"Placing rod {i+1} at mm:{PHANTOM_POINTS[i]} -> px:[{cx}, {cy}]")
        safe_add_patch(phantom, rod_patch, cx, cy)

    # 4. Save .pt File
    out_dict = {
        "Description": "Targeted Dual Hot Rod Phantom",
        "Metadata": {
            "size_mm": [fov_size_mm, fov_size_mm],
            "n_pixels": list(fov_n_pxs),
            "mm_per_pixel": [PX_SIZE, PX_SIZE],
            "rod_points_mm": PHANTOM_POINTS,
            "rod_diameter_mm": ROD_DIAMETER
        },
        "Phantom tensor": phantom
    }
    
    out_filename = f"targeted_phantom_{GRID_SIZE}px_{len(PHANTOM_POINTS)}p_{symnum}.pt"
    torch_save(out_dict, out_filename)
    print(f"Phantom saved to {out_filename}")

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # Use extent to map pixels to mm coordinates
    half_fov = fov_size_mm / 2
    extent = [-half_fov, half_fov, -half_fov, half_fov]
    
    im = ax.imshow(
        phantom.T, 
        cmap="gray_r", 
        extent=extent, 
        origin="lower", 
        interpolation="none"
    )
    
    plt.colorbar(im, label="Activity Intensity")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Targeted Phantom: 2 Rods (3mm Dia)\nLocations: {PHANTOM_POINTS}")
    
    # Zoom in to the center to see the rods clearly
    ax.set_xlim([-half_fov, half_fov])
    ax.set_ylim([-half_fov, half_fov])
    ax.grid(True, linestyle=":", alpha=0.4)

    plot_filename = out_filename.replace(".pt", ".png")
    fig.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    generate_targeted_phantom()