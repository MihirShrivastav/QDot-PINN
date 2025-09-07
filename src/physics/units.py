from __future__ import annotations

from dataclasses import dataclass
import math

# Fundamental constants (SI)
HBAR = 1.054_571_817e-34  # J*s
E_CHARGE = 1.602_176_634e-19  # C
EPS0 = 8.854_187_8128e-12  # F/m
M0 = 9.109_383_7015e-31  # kg

MEV_IN_J = 1.0e-3 * E_CHARGE  # 1 meV in Joules


@dataclass(frozen=True)
class GaAs:
    m_star_rel: float = 0.067  # m*/m0
    eps_r: float = 12.9
    L0_nm: float = 30.0

    @property
    def m_star(self) -> float:
        return self.m_star_rel * M0

    @property
    def L0(self) -> float:
        return self.L0_nm * 1e-9

    @property
    def E0_J(self) -> float:
        # E0 = hbar^2 / (2 m* L0^2)
        return (HBAR * HBAR) / (2.0 * self.m_star * self.L0 * self.L0)

    @property
    def E0_meV(self) -> float:
        return self.E0_J / MEV_IN_J

    @property
    def coulomb_gamma(self) -> float:
        # gamma = (e^2 / (4 pi eps L0)) / E0
        eps = self.eps_r * EPS0
        num = (E_CHARGE * E_CHARGE) / (4.0 * math.pi * eps * self.L0)
        return num / self.E0_J


def kappa_from_hbar_omega_meV(hbar_omega_meV: float, mat: GaAs) -> float:
    """
    Dimensionless harmonic curvature coefficient kappa for v(x) = kappa * x^2
    derived from target \hbar\omega (in meV) under the nondimensionalization with E0, L0.
    Formula: kappa = (m*^2 L0^4 / hbar^4) * (hbar*omega)^2
    """
    Eh_J = hbar_omega_meV * MEV_IN_J
    factor = (mat.m_star * mat.m_star) * (mat.L0 ** 4) / (HBAR ** 4)
    return factor * (Eh_J ** 2)


def pretty_summary(mat: GaAs) -> str:
    return (
        f"GaAs(m*/m0={mat.m_star_rel}, eps_r={mat.eps_r}, L0={mat.L0_nm} nm)\n"
        f"E0 ≈ {mat.E0_meV:.4f} meV, gamma ≈ {mat.coulomb_gamma:.3f}"
    )

