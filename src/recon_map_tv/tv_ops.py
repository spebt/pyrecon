"""
TV Gradient and Divergence Operators for MAP-TV Reconstruction.

Implements the 2D isotropic TV operators needed for the Chambolle-Pock
primal-dual solver. The divergence is the exact negative adjoint of the
gradient (backward difference), which is required for algorithm stability:
    <∇x, p> = -<x, div(p)>

For 3D extension: add a z-dimension term (Dz / Dz^T) to grad_2d and div_2d,
and update the stability constraint to τσ < 1/12 (from 1/8 for 2D).
"""

import torch


def grad_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D forward-difference gradient of image x.

    Args:
        x: Image tensor of shape (H, W).

    Returns:
        p: Gradient field of shape (2, H, W).
           p[0] = Dx (x-direction / columns), p[1] = Dy (y-direction / rows).

    Boundary: Zero-padding (Dirichlet) — last row/col difference is 0.
    """
    # Dx: forward difference along columns (dim=1)
    dx = torch.zeros_like(x)
    dx[:, :-1] = x[:, 1:] - x[:, :-1]  # interior; last column stays 0

    # Dy: forward difference along rows (dim=0)
    dy = torch.zeros_like(x)
    dy[:-1, :] = x[1:, :] - x[:-1, :]  # interior; last row stays 0

    return torch.stack([dx, dy], dim=0)  # (2, H, W)


def div_2d(p: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D backward-difference divergence of vector field p.
    This is the exact negative adjoint of grad_2d: div = -∇^T.

    Args:
        p: Gradient field of shape (2, H, W).
           p[0] = px (x-component), p[1] = py (y-component).

    Returns:
        d: Divergence of shape (H, W).
    """
    px, py = p[0], p[1]

    # Backward difference along columns (adjoint of Dx)
    # d_px[i, j] = px[i, j] - px[i, j-1]  with px[i, -1] = 0 at boundary
    d_px = torch.zeros_like(px)
    d_px[:, 0]  =  px[:, 0]           # first column: px[i,0] - 0
    d_px[:, 1:] =  px[:, 1:] - px[:, :-1]
    d_px[:, -1] = -px[:, -2]          # last column: 0 - px[i,-2]

    # Backward difference along rows (adjoint of Dy)
    d_py = torch.zeros_like(py)
    d_py[0, :]  =  py[0, :]           # first row
    d_py[1:, :] =  py[1:, :] - py[:-1, :]
    d_py[-1, :] = -py[-2, :]          # last row

    return d_px + d_py  # (H, W)


def tv_proximal_chambolle_pock(
    x_em: torch.Tensor,
    alpha: float,
    tau: float,
    sigma: float,
    theta: float,
    n_inner: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Solve the TV proximal problem via Chambolle-Pock primal-dual algorithm:
        x* = argmin_{x>=0}  0.5 * ||x - x_em||^2 + alpha * TV(x)

    Args:
        x_em:    EM-updated image, shape (H, W), on the correct device.
        alpha:   TV regularization strength (scaled beta).
        tau:     Primal step size. Stability: τσ < 1/8 for 2D.
        sigma:   Dual step size.
        theta:   Extrapolation factor (1.0 = standard Chambolle-Pock).
        n_inner: Number of inner iterations.
        eps:     Small floor to prevent division by zero.

    Returns:
        x: Denoised image of shape (H, W).
    """
    H, W = x_em.shape
    device = x_em.device

    # Initialize primal and dual variables
    x     = x_em.clone()
    x_bar = x_em.clone()
    p     = torch.zeros(2, H, W, device=device, dtype=x_em.dtype)  # dual field

    for _ in range(n_inner):
        x_prev = x.clone()

        # --- 1. Dual Update (gradient ascent + projection onto L∞/alpha ball) ---
        q = p + sigma * grad_2d(x_bar)                    # (2, H, W)
        # Pointwise L2 norm of the 2-vector at each pixel
        q_norm = torch.sqrt(q[0] ** 2 + q[1] ** 2).clamp(min=eps)  # (H, W)
        # Project: p = q / max(1, ||q|| / alpha)
        scale = torch.clamp(q_norm / alpha, min=1.0)      # (H, W)
        p = q / scale.unsqueeze(0)                        # (2, H, W)

        # --- 2. Primal Update (gradient descent + data fidelity proximal) ---
        # Closed-form solution to:  argmin_x 0.5*||x-x_em||^2 - tau*<div(p), x>
        x = (x_prev + tau * div_2d(p) + tau * x_em) / (1.0 + tau)
        x = torch.clamp(x, min=0.0)                       # non-negativity

        # --- 3. Extrapolation (over-relaxation) ---
        x_bar = x + theta * (x - x_prev)

    return x
