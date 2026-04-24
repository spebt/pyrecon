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


def grad_2d(x: torch.Tensor, out: torch.Tensor) -> None:
    """
    Write the 2D forward-difference gradient of x into pre-allocated `out`.

    Args:
        x:   Image tensor of shape (H, W).
        out: Pre-allocated buffer of shape (2, H, W). Must be zero-initialized
             once before first use — boundary positions (last col of out[0],
             last row of out[1]) are never written and must stay 0.

    Boundary: Zero-padding (Dirichlet) — last row/col difference is 0.
    """
    out[0, :, :-1] = x[:, 1:] - x[:, :-1]   # Dx interior; last col stays 0
    out[1, :-1, :] = x[1:, :] - x[:-1, :]   # Dy interior; last row stays 0


def div_2d(p: torch.Tensor, out: torch.Tensor) -> None:
    """
    Write the 2D backward-difference divergence of p into pre-allocated `out`.
    Exact negative adjoint of grad_2d: div = -∇^T.

    Args:
        p:   Gradient field of shape (2, H, W).
        out: Pre-allocated output buffer of shape (H, W). Fully overwritten.
    """
    px, py = p[0], p[1]

    # x-component (columns): write all positions cleanly without double-write
    out[:, 0]    =  px[:, 0]
    out[:, 1:-1] =  px[:, 1:-1] - px[:, :-2]
    out[:, -1]   = -px[:, -2]

    # y-component (rows): accumulate into out
    out[0, :]    +=  py[0, :]
    out[1:-1, :] +=  py[1:-1, :] - py[:-2, :]
    out[-1, :]   += -py[-2, :]


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

    All working buffers are pre-allocated before the loop — no heap
    allocations occur inside the inner iteration.

    Args:
        x_em:    EM-updated image, shape (H, W), on the correct device.
        alpha:   TV regularization strength (scaled beta).
        tau:     Primal step size. Stability: τσ < 1/8 for 2D.
        sigma:   Dual step size.
        theta:   Extrapolation factor (1.0 = standard Chambolle-Pock).
        n_inner: Number of inner iterations.
        eps:     Small floor to prevent division by zero in dual norm.

    Returns:
        x: Denoised image of shape (H, W).
    """
    H, W   = x_em.shape
    device = x_em.device
    dtype  = x_em.dtype

    # --- Pre-allocate all working buffers (zero allocs inside the loop) ---
    x        = x_em.clone()
    x_buf    = torch.empty_like(x)                        # primal buffer-swap partner
    x_bar    = x_em.clone()
    p        = torch.zeros(2, H, W, device=device, dtype=dtype)
    # grad_buf boundary positions (last col / last row) are written once as 0
    # and never touched again — Dirichlet BC is maintained across iterations.
    grad_buf = torch.zeros(2, H, W, device=device, dtype=dtype)
    div_buf  = torch.empty(H, W, device=device, dtype=dtype)
    norm_buf = torch.empty(H, W, device=device, dtype=dtype)  # dual norm / scale

    for _ in range(n_inner):
        # --- 1. Dual Update ---
        grad_2d(x_bar, out=grad_buf)
        p.add_(grad_buf, alpha=sigma)            # p += sigma * grad(x_bar)  [in-place]

        # Pointwise L2 norm of p, then reuse buffer as projection scale
        torch.mul(p[0], p[0], out=norm_buf)
        norm_buf.addcmul_(p[1], p[1])            # norm_buf = px^2 + py^2
        norm_buf.sqrt_().clamp_(min=eps)          # norm_buf = ||p||_2
        norm_buf.div_(alpha).clamp_(min=1.0)      # norm_buf = max(||p||/alpha, 1) = scale
        p.div_(norm_buf.unsqueeze(0))             # project p onto alpha-ball in-place

        # --- 2. Primal Update (buffer swap — no clone) ---
        # After swap: x_buf holds x^n (old iterate), x is the spare write target.
        x, x_buf = x_buf, x
        div_2d(p, out=div_buf)
        torch.add(x_buf, div_buf, alpha=tau, out=x)   # x = x_buf + tau * div(p)
        x.add_(x_em, alpha=tau)                        # x += tau * x_em
        x.div_(1.0 + tau)                              # x /= (1 + tau)
        x.clamp_(min=0.0)                              # non-negativity

        # --- 3. Extrapolation ---
        torch.sub(x, x_buf, out=x_bar)           # x_bar = x - x^n
        x_bar.mul_(theta).add_(x)                 # x_bar = x + theta*(x - x^n)

    return x
