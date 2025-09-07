from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from ..physics.units import GaAs, kappa_from_hbar_omega_meV


@dataclass
class BiquadraticParams:
    a: float  # half-separation (dimensionless)
    c4: float  # quartic coefficient
    c2y: float  # transverse harmonic coefficient
    delta: float = 0.0  # detuning (linear x term)


def v_biq_factory(p: BiquadraticParams) -> Callable[[float, float], float]:
    """Return v(x,y) for biquadratic potential (dimensionless)."""
    a2 = p.a * p.a

    def v(x: float, y: float) -> float:
        return p.c4 * ((x * x - a2) ** 2) + p.c2y * (y * y) + p.delta * x

    return v


def from_targets(
    a: float,
    hbar_omega_x_meV: float,
    hbar_omega_y_meV: float,
    delta: float = 0.0,
    mat: GaAs | None = None,
) -> BiquadraticParams:
    """
    Construct parameters from target single-dot curvatures (ħωx, ħωy in meV)
    at minima located at x=±a. Uses κx ≈ 4 c4 a^2 and κy = c2y.
    """
    mat = mat or GaAs()
    kx = kappa_from_hbar_omega_meV(hbar_omega_x_meV, mat)
    ky = kappa_from_hbar_omega_meV(hbar_omega_y_meV, mat)
    c4 = kx / (4.0 * a * a)
    c2y = ky
    return BiquadraticParams(a=a, c4=c4, c2y=c2y, delta=delta)


def suggest_default() -> Tuple[BiquadraticParams, GaAs]:
    mat = GaAs()  # L0=30 nm, m*/m0=0.067, eps_r=12.9
    params = from_targets(a=1.5, hbar_omega_x_meV=3.0, hbar_omega_y_meV=5.0, delta=0.0, mat=mat)
    return params, mat

