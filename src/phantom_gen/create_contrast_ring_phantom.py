import argparse
import math
import torch
from torch import float32 as torch_float32
from torch import save as torch_save
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.patches import Circle # Added for guide circles
import numpy as np

from _disk_shape import fov_tensor_dict, single_disk


# ---------- helpers ----------
def mm_to_px_vec2(v_mm: torch.Tensor, mm_per_px_xy: torch.Tensor, n_pixels_xy: torch.Tensor) -> torch.Tensor:
    """
    Convert mm coords (centered at (0,0)) to pixel indices in [0..N-1]^2.
    """
    return torch.round(v_mm / mm_per_px_xy + n_pixels_xy.to(v_mm.dtype) * 0.5).to(torch.long)

def safe_add_patch(img: torch.Tensor, patch: torch.Tensor, cx: int, cy: int) -> None:
    """
    Add 'patch' (ph x pw) into 'img' (H x W) centered at (cx, cy).
    Uses EXCLUSIVE end indices to avoid off-by-one with even-sized patches.
    Handles boundary clipping.
    """
    H, W = img.shape
    ph, pw = patch.shape

    x0 = cx - ph // 2
    y0 = cy - pw // 2
    x1 = x0 + ph
    y1 = y0 + pw

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(H, x1)
    sy1 = min(W, y1)
    if sx0 >= sx1 or sy0 >= sy1:
        return

    px0 = sx0 - x0
    py0 = sy0 - y0
    px1 = px0 + (sx1 - sx0)
    py1 = py0 + (sy1 - sy0)

    img[sx0:sx1, sy0:sy1] += patch[px0:px1, py0:py1]


def build_ring_hotrods_phantom(
    fov_dict: Dict[str, torch.Tensor],
    n_rods: int,
    rod_diameter_mm: float,
    ring_inner_radius_mm: float,
    ring_outer_radius_mm: float,
    bg_counts: float = 5.0,
    rod_counts: float = 20.0,
    supersample_factor: int = 16,
) -> Dict:
    H, W = map(int, fov_dict["n pixels"].tolist())
    mm_per_px = fov_dict["mm per pixel"]
    n_pixels_xy = fov_dict["n pixels"]
    phantom = torch.full((H, W), float(bg_counts), dtype=torch_float32)

    # Calculate radius in pixels (int)
    r_px_val = max(1, int(round((rod_diameter_mm * 0.5) / float(mm_per_px[0]))))
    # FIX: Convert r_px to a Tensor for single_disk
    r_px_tensor = torch.tensor(r_px_val, dtype=torch_float32)
    disk = single_disk(r_px_tensor, factor=supersample_factor)
    
    # Keep integer for metadata
    r_px = r_px_val 

    R_in, R_out = ring_inner_radius_mm, ring_outer_radius_mm
    R_mid = 0.5 * (R_in + R_out)
    rod_radius_mm = 0.5 * rod_diameter_mm

    # Safety: radial clearance
    radial_clearance = min(R_mid - R_in, R_out - R_mid)
    assert radial_clearance >= rod_radius_mm, (
        f"Rod radius ({rod_radius_mm} mm) exceeds radial clearance ({radial_clearance} mm). "
        f"Increase ring width or reduce rod diameter."
    )

    # Angular spacing check
    arc_per_rod = 2.0 * math.pi * R_mid / n_rods
    if arc_per_rod < 2.2 * rod_radius_mm:
        print("[warn] Angular spacing is very tight; rods may appear close/overlapping.")

    # Place rods at equally spaced angles
    angles = torch.linspace(0.0, 2 * math.pi, steps=n_rods + 1, dtype=torch.float32)[:-1]
    centers_mm = torch.stack(
        (R_mid * torch.cos(angles), R_mid * torch.sin(angles)),
        dim=1,
    )

    centers_px = mm_to_px_vec2(centers_mm, mm_per_px, n_pixels_xy)
    delta = float(rod_counts - bg_counts)
    patch = disk * delta

    rods_info: List[Dict] = []
    for k in range(n_rods):
        cx, cy = int(centers_px[k, 0].item()), int(centers_px[k, 1].item())
        safe_add_patch(phantom, patch, cx, cy)
        rods_info.append(
            {"center_px": (cx, cy), "radius_px": r_px, "diameter_mm": rod_diameter_mm}
        )

    return {
        "phantom": phantom,
        "rods": rods_info,
        "metadata": {
            "layout": "ring",
            "n_rods": int(n_rods),
            "rod_diameter_mm": float(rod_diameter_mm),
            "ring_inner_radius_mm": float(R_in),
            "ring_outer_radius_mm": float(R_out),
            "bg_counts": float(bg_counts),
            "rod_counts": float(rod_counts),
            "mm_per_pixel": fov_dict["mm per pixel"].tolist(),
            "size_in_mm": fov_dict["size in mm"].tolist(),
            "n_pixels": fov_dict["n pixels"].tolist(),
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Create a ring phantom with hot rods.")
    p.add_argument("--size-mm", type=float, nargs=2, default=(128.0, 128.0))
    p.add_argument("--px-mm",   type=float, nargs=2, default=(0.25, 0.25))

    # Ring + rods
    p.add_argument("--n-rods", type=int, default=24, help="Number of rods on the ring")
    p.add_argument("--rod-diameter-mm", type=float, default=3.0, help="Single rod diameter (mm)")
    
    # UPDATED DEFAULTS: 15.0 and 25.0 are the script's defaults. 
    # To match your other script, run with: --ring-inner-mm 20.0 --ring-outer-mm 25.0
    p.add_argument("--ring-inner-mm", type=float, default=20.0, help="Inner ring radius (mm)")
    p.add_argument("--ring-outer-mm", type=float, default=25.0, help="Outer ring radius (mm)")

    # Counts
    p.add_argument("--bg-counts", type=float, default=5.0, help="Background counts per pixel")
    p.add_argument("--rod-counts", type=float, default=20.0, help="Hot-rod counts per pixel")

    p.add_argument("--ss-factor", type=int, default=16, help="Supersampling factor")
    p.add_argument("--out-prefix", type=str, default="ring_hotrods_phantom")
    return p.parse_args()


def main():
    args = parse_args()

    n_px_x = int(round(args.size_mm[0] / args.px_mm[0]))
    n_px_y = int(round(args.size_mm[1] / args.px_mm[1]))
    fov = fov_tensor_dict(
        n_pixels=(n_px_x, n_px_y),
        size_in_mm=(args.size_mm[0], args.size_mm[1]),
        center_coordinates=(0.0, 0.0),
    )

    pack = build_ring_hotrods_phantom(
        fov_dict=fov,
        n_rods=args.n_rods,
        rod_diameter_mm=args.rod_diameter_mm,
        ring_inner_radius_mm=args.ring_inner_mm,
        ring_outer_radius_mm=args.ring_outer_mm,
        bg_counts=args.bg_counts,
        rod_counts=args.rod_counts,
        supersample_factor=args.ss_factor,
    )

    phantom = pack["phantom"]
    out_dict = {
        "Description": "Ring phantom with hot rods",
        "Metadata": pack["metadata"],
        "Phantom tensor": phantom,
        "Rods": pack["rods"],
        "Phantom shape": phantom.shape,
        "Phantom dtype": phantom.dtype,
    }

    # Save .pt
    pt_name = f"{args.out_prefix}_{int(args.size_mm[0])}x{int(args.size_mm[1])}mm_{n_px_x}x{n_px_y}px.pt"
    torch_save(out_dict, pt_name)
    print(f"Saved: {pt_name}")

    # --- Plotting with MM Axes ---
    plot_name = pt_name.replace(".pt", ".png")
    phantom_np = phantom.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extent logic: [left, right, bottom, top] in mm
    # Assumes center is (0,0)
    half_x = args.size_mm[0] / 2.0
    half_y = args.size_mm[1] / 2.0
    extent_mm = [-half_x, half_x, -half_y, half_y]

    # Plot using gray_r (white background, dark spots)
    # Transpose (.T) so X-axis corresponds to dim0, Y-axis to dim1 (standard cartesian)
    im = ax.imshow(
        phantom_np.T, 
        cmap='gray_r', 
        extent=extent_mm, 
        origin='lower'
    ) 
    
    # Add Colorbar
    plt.colorbar(im, ax=ax, label='Counts')

    # Add Guide Circles (Red) to verify geometry
    inner_circ = Circle((0, 0), args.ring_inner_mm, color='red', fill=False, linestyle='--', alpha=0.5, label='Inner Ring')
    outer_circ = Circle((0, 0), args.ring_outer_mm, color='red', fill=False, linestyle='--', alpha=0.5, label='Outer Ring')
    ax.add_patch(inner_circ)
    ax.add_patch(outer_circ)

    # Labels and Title
    ax.set_title(f"Ring Hot-Rods (Inner={args.ring_inner_mm}mm, Outer={args.ring_outer_mm}mm)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    
    # Force aspect ratio to be equal so circles look like circles
    ax.set_aspect('equal')
    
    # Optional legend for the rings
    ax.legend(loc='upper right')

    plt.savefig(plot_name, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {plot_name} [Image with MM axes]")

if __name__ == "__main__":
    main()