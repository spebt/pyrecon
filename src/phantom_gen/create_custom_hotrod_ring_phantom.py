import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import sys

# --- Helper Functions (inspired by your provided code) ---

def fov_tensor_dict(n_pixels, fov_size_in_mm, center_coordinates=(0.0, 0.0)):
    """
    Creates a dictionary with FOV parameters as tensors.
    """
    fov_n_pxs = torch.tensor(n_pixels)
    fov_size_mm = torch.tensor(fov_size_in_mm)
    mm_per_pixel = fov_size_mm / fov_n_pxs
    center_coords_mm = torch.tensor(center_coordinates)

    return {
        "n pixels": fov_n_pxs,
        "size in mm": fov_size_mm,
        "mm per pixel": mm_per_pixel,
        "center coordinates in mm": center_coords_mm,
    }

def add_disk_to_phantom(phantom, center_mm, radius_mm, fov_dict):
    """
    Adds a disk (rod) with a value of 1.0 to the phantom tensor.
    
    Note: Assumes phantom is (X, Y) and uses 'ij' indexing.
    """
    n_px = fov_dict["n pixels"]
    size_mm = fov_dict["size in mm"]
    
    # Create pixel coordinate vectors for X and Y
    x_coords_mm = torch.linspace(-size_mm[0] / 2.0, size_mm[0] / 2.0, int(n_px[0]))
    y_coords_mm = torch.linspace(-size_mm[1] / 2.0, size_mm[1] / 2.0, int(n_px[1]))
    
    # Create coordinate grids (using 'ij' indexing to match (X, Y) tensor)
    xx, yy = torch.meshgrid(x_coords_mm, y_coords_mm, indexing='ij')
    
    # Calculate squared distance from the rod's center
    dist_sq = (xx - center_mm[0])**2 + (yy - center_mm[1])**2
    radius_sq = radius_mm**2
    
    # Create the mask
    mask = (dist_sq <= radius_sq)
    
    # Apply the mask to the phantom
    phantom[mask] = 1.0

# --- Main Script ---

if __name__ == "__main__":
    
    # --- Parameter Handling ---
    if len(sys.argv) != 3:
        print("Usage: python create_ring_phantom.py <num_rods> <rod_diameter_mm>")
        print("Example: python create_ring_phantom.py 12 3.0")
        sys.exit(1)
        
    try:
        NUM_RODS = int(sys.argv[1])
        ROD_DIAMETER_MM = float(sys.argv[2])
    except ValueError:
        print("Error: <num_rods> must be an integer and <rod_diameter_mm> must be a float.")
        sys.exit(1)

    print(f"--- Generating Ring Phantom ---")
    print(f"  Number of Rods: {NUM_RODS}")
    print(f"  Rod Diameter:   {ROD_DIAMETER_MM} mm")

    # --- Phantom Configuration ---
    FOV_SIZE_MM = (128.0, 128.0)
    FOV_N_PX = (512, 512)
    
    INNER_DIAMETER_MM = 40.0
    OUTER_DIAMETER_MM = 50.0

    # --- Derived Calculations ---
    ROD_RADIUS_MM = ROD_DIAMETER_MM / 2.0
    INNER_RADIUS_MM = INNER_DIAMETER_MM / 2.0
    OUTER_RADIUS_MM = OUTER_DIAMETER_MM / 2.0
    
    # Place rods at the midpoint radius between the two circles
    PLACEMENT_RADIUS_MM = (INNER_RADIUS_MM + OUTER_RADIUS_MM) / 2.0 
    
    # Setup FOV dictionary
    fov_dict = fov_tensor_dict(FOV_N_PX, FOV_SIZE_MM)
    
    # Initialize empty phantom tensor (X, Y)
    phantom = torch.zeros(FOV_N_PX, dtype=torch.float32)

    # --- Calculate Rod Center Coordinates ---
    # Generate equally spaced angles
    angles = torch.linspace(0, 2 * torch.pi, NUM_RODS + 1)[:-1]
    
    # Convert angles to (x, y) coordinates
    x_centers_mm = PLACEMENT_RADIUS_MM * torch.cos(angles)
    y_centers_mm = PLACEMENT_RADIUS_MM * torch.sin(angles)
    rod_centers_mm = torch.stack([x_centers_mm, y_centers_mm], dim=1)

    print(f"  Placement Radius: {PLACEMENT_RADIUS_MM:.2f} mm")

    # --- Generate Phantom ---
    print("Drawing rods onto phantom tensor...")
    for center_mm in rod_centers_mm:
        add_disk_to_phantom(phantom, center_mm, ROD_RADIUS_MM, fov_dict)

    # --- Save Phantom Tensor (.pt file) ---
    out_dict = {
        "Description": "Ring Hot Rods Phantom",
        "Metadata": {
            "size in mm": fov_dict["size in mm"].tolist(),
            "mm per pixel": fov_dict["mm per pixel"].tolist(),
            "n pixels": fov_dict["n pixels"].tolist(),
            "center coordinates in mm": fov_dict["center coordinates in mm"].tolist(),
            "num_rods": NUM_RODS,
            "rod_diameter_mm": ROD_DIAMETER_MM,
            "inner_guide_diameter_mm": INNER_DIAMETER_MM,
            "outer_guide_diameter_mm": OUTER_DIAMETER_MM,
        },
        "Phantom tensor": phantom,
        "Phantom shape": phantom.shape,
        "Phantom dtype": phantom.dtype,
    }
    
    pt_filename = f"ring_phantom_{NUM_RODS}rods_{ROD_DIAMETER_MM}mm.pt"
    torch.save(out_dict, pt_filename)
    print(f"Phantom tensor saved to: {pt_filename}")

    # --- Plot Phantom Image (.png file) ---
    print("Generating plot...")
    plt.close("all")
    fig, ax = plt.subplots(dpi=150, figsize=(10, 10), layout="constrained")
    
    # Plot the phantom (use .T to swap X,Y for imshow's Y,X)
    ax.imshow(
        phantom.T,
        cmap="gray_r",
        extent=(
            -FOV_SIZE_MM[0] / 2,
            FOV_SIZE_MM[0] / 2,
            -FOV_SIZE_MM[1] / 2,
            FOV_SIZE_MM[1] / 2,
        ),
        origin="lower",
        aspect="equal",
        interpolation="none",
    )
    
    # Add the inner and outer guide circles
    inner_circle = Circle((0, 0), INNER_RADIUS_MM, color='red', fill=False, lw=1)
    outer_circle = Circle((0, 0), OUTER_RADIUS_MM, color='red', fill=False, lw=1)
    ax.add_artist(inner_circle)
    ax.add_artist(outer_circle)
    
    # Set labels and title
    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)
    ax.set_xticks(torch.arange(-60, 61, 10))
    ax.set_yticks(torch.arange(-60, 61, 10))
    ax.set_title(f"Ring Phantom ({NUM_RODS} Rods @ {ROD_DIAMETER_MM}mm Diameter)", fontsize=14)
    
    # Save the figure
    png_filename = f"ring_phantom_{NUM_RODS}rods_{ROD_DIAMETER_MM}mm.png"
    fig.savefig(png_filename, dpi=150)
    print(f"Phantom plot saved to: {png_filename}")
    print("--- Done ---")