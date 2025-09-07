from __future__ import annotations

from typing import Callable, Tuple

import torch


def laplacian(fn: Callable[[torch.Tensor], torch.Tensor], xy: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇^2 ψ for ψ=fn(xy), xy shape [N,2], using autograd.
    """
    with torch.enable_grad():
        xy = xy.detach().requires_grad_(True)
        psi = fn(xy)  # [N,1]
        grads = torch.autograd.grad(psi, xy, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        dpsi_dx = grads[:, 0:1]
        dpsi_dy = grads[:, 1:2]
        d2psi_dx2 = torch.autograd.grad(dpsi_dx, xy, grad_outputs=torch.ones_like(dpsi_dx), create_graph=True, retain_graph=True)[0][:, 0:1]
        d2psi_dy2 = torch.autograd.grad(dpsi_dy, xy, grad_outputs=torch.ones_like(dpsi_dy), create_graph=True, retain_graph=True)[0][:, 1:2]
        return d2psi_dx2 + d2psi_dy2


def h_psi(fn: Callable[[torch.Tensor], torch.Tensor], vfun: Callable[[torch.Tensor], torch.Tensor], xy: torch.Tensor) -> torch.Tensor:
    """Apply dimensionless Hamiltonian H = -∇^2 + v(x,y) to ψ."""
    with torch.enable_grad():
        psi = fn(xy)
        lap = laplacian(fn, xy)
        v = vfun(xy)
        return -lap + v * psi


def rayleigh_ritz_energy(fn: Callable[[torch.Tensor], torch.Tensor], vfun: Callable[[torch.Tensor], torch.Tensor], xy_q: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    """
    E(ψ) = [∫ (|∇ψ|² + v |ψ|²) dΩ] / [∫ |ψ|² dΩ]
    Using grad-squared form for kinetic term ensures positivity without relying on boundary terms.
    """
    with torch.enable_grad():
        # Ensure input is a leaf with requires_grad for autograd.grad
        xy_leaf = xy_q.detach().requires_grad_(True)
        psi = fn(xy_leaf)  # [N,1]
        grads = torch.autograd.grad(psi, xy_leaf, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        grad_sq = torch.sum(grads * grads, dim=1, keepdim=True)  # [N,1]
        v = vfun(xy_leaf)
        num = torch.sum(w_q * (grad_sq + v * psi * psi))
        den = torch.sum(w_q * psi * psi)
        return num / (den + 1e-12)


def pde_residual_loss(fn: Callable[[torch.Tensor], torch.Tensor], vfun: Callable[[torch.Tensor], torch.Tensor], xy_c: torch.Tensor) -> torch.Tensor:
    psi = fn(xy_c)
    Hpsi = h_psi(fn, vfun, xy_c)
    e_local = torch.sum(psi * Hpsi, dim=0, keepdim=True) / (torch.sum(psi * psi, dim=0, keepdim=True) + 1e-12)
    res = Hpsi - e_local * psi
    return torch.mean(res * res)


def orthogonality_loss(psi_new: torch.Tensor, psi_prev: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    # assumes shapes [N,1]
    num = torch.sum(w_q * psi_new * psi_prev)
    den = torch.sqrt(torch.sum(w_q * psi_new * psi_new) * torch.sum(w_q * psi_prev * psi_prev) + 1e-12)
    return (num / (den + 1e-12)) ** 2

