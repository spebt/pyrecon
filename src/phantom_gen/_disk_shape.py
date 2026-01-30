import math
from typing import Dict, Sequence, Tuple

from torch import (
    Tensor,
    arange,
    cat,
    meshgrid,
    norm,
    ones,
    sqrt,
    stack,
    tensor,
    zeros,
    bucketize,
    ceil,
)
from torch import float32 as torch_float32
from torch.nn.functional import conv2d

# Assuming _geometry_2d_transform.py is in the same directory
from _geometry_2d_transform import transform_to_positions_2d_batch


def fov_tensor_dict(
    n_pixels: Sequence[int] = (512, 512),
    size_in_mm: Sequence[float] = (128, 128),
    center_coordinates: Sequence[float] = (0.0, 0.0),
    n_subdivisions: Sequence[int] = (1, 1),
) -> dict:
    """
    Create a dictionary with the FOV information.
    """
    fov_dict = {
        "n pixels": tensor(n_pixels),
        "size in mm": tensor(size_in_mm),
        "center coordinates in mm": tensor(center_coordinates),
    }
    fov_dict["mm per pixel"] = fov_dict["size in mm"] / fov_dict["n pixels"]
    fov_dict["n subdivisions"] = tensor(n_subdivisions)
    return fov_dict


def high_res_disk_mask(
    high_res_radius: float, high_res_mask_size_half: int
) -> Tensor:
    """
    Create a high-resolution binary disk mask.

    Parameters
    ----------
    high_res_radius : float
        The precise radius of the disk in high-resolution pixels.
    high_res_mask_size_half : int
        The half-width of the mask to be generated (in high-res pixels).
        The final mask will be (2 * high_res_mask_size_half, 2 * high_res_mask_size_half).
    """
    epsilon = 1e-6
    mask_size = high_res_mask_size_half * 2
    mask = zeros((mask_size, mask_size), dtype=torch_float32)

    # Create a grid of pixel indices
    pixel_grid = stack(
        meshgrid(
            arange(mask_size + 1),
            arange(mask_size + 1),
            indexing="ij",
        ),
        dim=-1,
    )

    # Get centroids and corners for all high-res pixels
    pixel_centroids = (
        (pixel_grid[:-1, :-1].float() + 0.5) - high_res_mask_size_half
    )
    pixel_four_corners = (
        stack(
            (
                pixel_grid[:-1:, :-1],
                pixel_grid[1:, :-1],
                pixel_grid[1:, 1:],
                pixel_grid[:-1, 1:],
            ),
            dim=-2,
        ).view(-1, 4, 2)
        - high_res_mask_size_half
    )
    
    # Combine centroids and corners (5 points per pixel)
    pixel_five_points = cat(
        (pixel_four_corners, pixel_centroids.view(-1, 1, 2)), dim=1
    )

    # Check distance of all 5 points from the center
    distance_from_center = norm(pixel_five_points, dim=-1)
    
    # A pixel is considered "in" if its centroid is in the circle
    # This is a simple but effective supersampling method
    mask.view(-1)[distance_from_center[:, 4] < high_res_radius - epsilon] = 1.0
    return mask


def down_sample_disk_mask(mask: Tensor, downsampling_factor: int) -> Tensor:
    """
    Down-sample a high-resolution disk mask by a given factor
    using average pooling (convolution).
    """
    return conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        ones(
            (1, 1, downsampling_factor, downsampling_factor),
            dtype=torch_float32,
        )
        / (downsampling_factor**2),
        padding=0,
        stride=downsampling_factor,
    ).squeeze()


def single_disk(radius_px: float, factor: int = 16) -> Tensor:
    """
    Create a single anti-aliased disk mask with a given float radius.
    """
    # Calculate the integer half-width of the final mask in pixels
    # The mask will be (2 * radius_px_int, 2 * radius_px_int)
    radius_px_int = int(ceil(radius_px))
    
    # Avoid creating empty tensors if radius is too small
    if radius_px_int == 0:
        return zeros((0, 0), dtype=torch_float32)

    # Calculate the radius and mask size in high-resolution pixels
    high_res_radius = radius_px * factor
    high_res_mask_size_half = radius_px_int * factor

    # Create the high-resolution binary mask
    mask = high_res_disk_mask(high_res_radius, high_res_mask_size_half)
    
    # Down-sample to get the final anti-aliased (partial volume) mask
    return down_sample_disk_mask(mask, factor)


def hot_rods_sector_centers(
    n_x_layers: int,
    radius: float,
    transform: Tensor,
    fov_dict: Dict[str, Tensor],
    rod_spacing_factor: float = 2.0,
) -> Tuple[Tensor, Tensor]:
    """
    Get the centers of the sectors in a hexagonal grid.

    Parameters
    ----------
    n_x_layers : int
        Number of layers in the x-direction.
    radius : float
        Radius of the rods in millimeters.
    transform : torch.Tensor
        Transformation (rotation, shift) to apply. Shape (1, 3).
    fov_dict : Dict[str, Tensor]
        Dictionary containing the field of view information.
    rod_spacing_factor : float, optional
        Multiplier for the center-to-center distance.
        1.0 = touching (center-dist = 2 * radius).
        2.0 = one diameter gap (center-dist = 4 * radius).
        Default is 2.0 (matches original script).

    Returns
    -------
    sector_centers_mm : torch.Tensor
        Centers of the sectors in millimeters.
    sector_centers_px : torch.Tensor
        Centers of the sectors in pixels (as indices).
    """

    # This grid has a minimum center-to-center distance of 2.0
    sector_center_grid = stack(
        [
            tensor([i for j in range(0, n_x_layers) for i in [j] * (j + 1)])
            * sqrt(tensor([3.0])),
            tensor(
                [
                    ia + ib * 2
                    for j in range(0, n_x_layers)
                    for ia, ib in zip([j] * (j + 1), range(-j, j + 1, 1))
                ]
            ),
        ],
        dim=1,
    )

    # Scale the grid.
    # A factor of `radius` makes the center-dist 2*radius (touching).
    # The `rod_spacing_factor` scales this.
    sector_centers_mm = sector_center_grid * radius * rod_spacing_factor
    
    # Apply 2D transformation (rotation and shift)
    shift_before = transform[:, 1:]
    sector_centers_mm = sector_centers_mm + shift_before.squeeze()
    transform_rot_only = cat(
        (transform[:, :1], tensor([[0.0, 0.0]])), dim=1
    )
    sector_centers_mm = transform_to_positions_2d_batch(
        transform_rot_only,
        sector_centers_mm,
    ).view(-1, 2)
    
    # Get pixel indices from mm coordinates
    # We create boundaries for bucketization
    boundaries_x = (
        arange(
            -int(fov_dict["n pixels"][0] / 2), int(fov_dict["n pixels"][0] / 2) + 1
        )
        * fov_dict["mm per pixel"][0]
        + fov_dict["center coordinates in mm"][0]
    )
    boundaries_y = (
        arange(
            -int(fov_dict["n pixels"][1] / 2), int(fov_dict["n pixels"][1] / 2) + 1
        )
        * fov_dict["mm per pixel"][1]
        + fov_dict["center coordinates in mm"][1]
    )
    
    # Bucketize finds the index for each mm coordinate
    sector_centers_px_x = bucketize(sector_centers_mm[:, 0], boundaries_x) - 1
    sector_centers_px_y = bucketize(sector_centers_mm[:, 1], boundaries_y) - 1
    
    sector_centers_px = stack((sector_centers_px_x, sector_centers_px_y), dim=1)
    
    return sector_centers_mm, sector_centers_px


def hot_rods_add_sector(
    phantom: Tensor,
    n_x_layers: int,
    radius: float,
    transform: Tensor,
    fov_dict: Dict[str, Tensor],
    rod_spacing_factor: float = 2.0,
):
    """
    Add a sector of hot rods to the phantom.
    """
    # Calculate radius in pixels as a float
    pxs_per_mm = 1.0 / fov_dict["mm per pixel"][0]
    radius_px = radius * pxs_per_mm  # This is now a float

    # Get rod centers
    sector_centers_mm, sector_centers_px = hot_rods_sector_centers(
        n_x_layers, radius, transform, fov_dict, rod_spacing_factor
    )
    
    # Create one anti-aliased disk
    disk = single_disk(radius_px)
    
    if disk.nelement() == 0:
        # Disk is too small to be drawn
        return sector_centers_mm, sector_centers_px

    disk_half_width = disk.shape[0] // 2
    n_px_x, n_px_y = phantom.shape

    for center_px in sector_centers_px:
        center_x_idx = int(center_px[0])
        center_y_idx = int(center_px[1])

        # --- Robust Slicing (Clipping) ---
        # Calculate phantom slice indices (clamped to FOV)
        x_start_phantom = max(0, center_x_idx - disk_half_width)
        x_end_phantom = min(n_px_x, center_x_idx + disk_half_width)
        y_start_phantom = max(0, center_y_idx - disk_half_width)
        y_end_phantom = min(n_px_y, center_y_idx + disk_half_width)

        # Calculate corresponding disk slice indices
        x_start_disk = max(0, disk_half_width - center_x_idx)
        x_end_disk = disk.shape[0] - max(
            0, (center_x_idx + disk_half_width) - n_px_x
        )
        y_start_disk = max(0, disk_half_width - center_y_idx)
        y_end_disk = disk.shape[1] - max(
            0, (center_y_idx + disk_half_width) - n_px_y
        )
        # ----------------------------------

        # Add disk to phantom if it's within bounds
        if (
            x_start_phantom < x_end_phantom
            and y_start_phantom < y_end_phantom
        ):
            phantom[
                x_start_phantom:x_end_phantom,
                y_start_phantom:y_end_phantom,
            ] += disk[
                x_start_disk:x_end_disk,
                y_start_disk:y_end_disk,
            ]
            
    return sector_centers_mm, sector_centers_px