import os
import sys
from typing import Dict, List
import torch
from matplotlib import pyplot as plt
from torch import (
    arange, cat, cos, ones, pi, sin, tensor, zeros, 
    save as torch_save, load as torch_load
)
from torch import float32 as torch_float32

# Note: Ensure _disk_shape.py is in your PYTHONPATH
from _disk_shape import fov_tensor_dict, hot_rods_add_sector

# =============================================================================
# 1. FIELD OF VIEW (FOV) & GRID INITIALIZATION
# =============================================================================

# Spatial dimensions of the phantom grid in millimeters
fov_size_in_mm = (128.0, 128.0) 

# Resolution of the phantom grid
fov_px_size_in_mm = (0.25, 0.25)

# Calculate total number of pixels based on physical size and resolution
fov_n_pxs = (
    (tensor(fov_size_in_mm) / tensor(fov_px_size_in_mm)).int().tolist()
)

# Initialize FOV metadata and empty phantom tensor
fov_dict = fov_tensor_dict(fov_n_pxs, fov_size_in_mm)
phantom = zeros(fov_n_pxs, dtype=torch_float32)


# =============================================================================
# 2. PHANTOM GEOMETRY & SECTOR DEFINITIONS
# =============================================================================

# Rod radii for each of the 6 sectors (mm)
radii = tensor([0.5, 1, 1.5, 2, 2.5, 3])

# Translation shifts from FOV center for each sector [x, y] in mm
shifts = tensor([
    [10, 0.0], [10, 0.0], [10, 0.0], [10, 0.0], [12, 0.0], [12, 0.0]
])

# Density of rods: number of layers in the radial direction for each sector
n_x_layers = tensor([20, 12, 8, 6, 5, 4])

# Rod Spacing Multiplier:
# 1.0 = Rods are touching (Center-to-Center distance = 2 * radius)
# 2.0 = Gap equals rod diameter (Center-to-Center distance = 4 * radius)
rod_spacing_factors = tensor([2, 2, 2, 2, 2, 2])

# Angular orientation for each sector (60-degree increments)
angles = arange(0, 2 * pi, 2 * pi / 6).unsqueeze(-1) + pi / 6


# =============================================================================
# 3. PHANTOM GENERATION LOOP
# =============================================================================

print("Generating phantom...")
sectors_centers_mm = []

for i in range(shifts.shape[0]):
    print(
        f"  Adding sector {i+1}: "
        f"radius={radii[i].item():.1f}mm, "
        f"spacing_factor={rod_spacing_factors[i].item()}"
    )
    
    # Combine rotation and translation into a single transform tensor
    transform = cat((angles[i : i + 1], shifts[i : i + 1]), dim=1)
    
    # Generate rods for the current sector and update the phantom tensor in-place
    sector_centers_mm, sector_centers_px = hot_rods_add_sector(
        phantom,
        int(n_x_layers[i].item()),
        radii[i].item(),
        transform,
        fov_dict,
        rod_spacing_factors[i].item(),
    )
    sectors_centers_mm.append(sector_centers_mm)

print("Phantom generation complete.")


# =============================================================================
# 4. DATA SERIALIZATION
# =============================================================================

out_dict = {
    "Description": "Hot Rods Phantom",
    "Metadata": {
        "size in mm": fov_dict["size in mm"].tolist(),
        "mm per pixel": fov_dict["mm per pixel"].tolist(),
        "n pixels": fov_dict["n pixels"].tolist(),
        "center coordinates in mm": fov_dict["center coordinates in mm"].tolist(),
        "rods radii in mm": radii.tolist(),
        "rod spacing factors": rod_spacing_factors.tolist(),
    },
    "Phantom tensor": phantom,
    "Phantom shape": phantom.shape,
    "Phantom dtype": phantom.dtype,
}

out_filename = f'hot_rods_phantom_{fov_dict["size in mm"][0].item()}_mm_x_{fov_dict["size in mm"][1].item()}_mm.pt'
torch_save(out_dict, out_filename)
print(f"Phantom saved to {out_filename}")


# =============================================================================
# 5. VISUALIZATION & PLOTTING
# =============================================================================

print("Generating plot...")
if not os.path.isfile(out_filename):
    print(f"File {out_filename} does not exist.")
    sys.exit(1)

try:
    data = torch_load(out_filename, map_location="cpu")
except Exception as e:
    print(f"Error reading file {out_filename}: {e}")
    sys.exit(1)

# Extract plot parameters from loaded metadata
radii = tensor(data["Metadata"]["rods radii in mm"])
base_angles = arange(0, 2 * 3.14159, 2 * 3.14159 / 6).unsqueeze(-1) + 3.14159 / 6

# Adjust annotation angles to align text with the visual orientation of sectors
anno_angles = base_angles + (3 * 3.14159 / 2) 
anno_radii = ones(6) * 30.25  # Distance from center for text labels (mm)

# Compute Cartesian coordinates for annotation text
anno_xy = cat(
    (
        cos(anno_angles) * anno_radii.view(-1, 1),
        sin(anno_angles) * anno_radii.view(-1, 1),
    ),
    -1,
)
anno_text = [f"{r*2:.2f} mm" for r in radii]

# Plot Setup
plt.close("all")
fig, ax = plt.subplots(dpi=150, figsize=(12, 12), layout="constrained")

# Rotate phantom 270 degrees to match standard viewing orientation
phantom_rotated = torch.rot90(data["Phantom tensor"], k=3, dims=(0, 1))

ax.imshow(
    phantom_rotated.T,
    cmap="gray_r",
    extent=(
        -fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2,
        -fov_size_in_mm[1] / 2, fov_size_in_mm[1] / 2,
    ),
    origin="lower",
    aspect="equal",
    interpolation="none",
    vmin=0,
    vmax=phantom.max()
)

# Axis Ticks and Labels
tick_spacing = 20
x_ticks = arange(-int(fov_size_in_mm[0]/2), int(fov_size_in_mm[0]/2)+1, tick_spacing)
y_ticks = arange(-int(fov_size_in_mm[1]/2), int(fov_size_in_mm[1]/2)+1, tick_spacing)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.5)

# Place diameter annotations for each sector
for i in range(len(anno_text)):
    ax.annotate(
        anno_text[i],
        xy=(0, 0),
        xytext=tuple(anno_xy[i].tolist()),
        textcoords="data",
        fontsize=20,
        ha="center",
        va="center",
        color="w",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.5),
    )

# Final Title generation
metadata = data["Metadata"]
fig_title = (
    f"Hot Rods Phantom, {metadata['size in mm'][0]} mm X {metadata['size in mm'][1]} mm\n"
    f"{metadata['n pixels'][0]} px X {metadata['n pixels'][1]} px "
    f"({metadata['mm per pixel'][0]:.4f} mm/px)"
)
fig.suptitle(fig_title, fontsize=20)

# Save visualization
out_plot_filename = out_filename.replace(".pt", ".png")
fig.savefig(out_plot_filename, dpi=150)
print(f"Plot saved to {out_plot_filename}")