from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
from physics.units import GaAs

@dataclass
class GaussianParams:
    """
    Stores dimensionless parameters for a double Gaussian potential.
    
    The potential is scaled such that the well minima are at V≈0,
    and the potential at infinity is V≈v0. This ensures all potential
    values are non-negative, which helps stabilize PINN training.
    """
    a: float        # Half-separation between wells
    v0: float       # Well depth, which also serves as the potential at infinity
    sigma: float    # Well width (Gaussian standard deviation)
    v_b: float      # Additional height of the central barrier *above* the baseline
    sigma_b: float  # Width of the central barrier
    delta: float = 0.0  # Detuning term to tilt the potential

def v_gaussian_factory(p: GaussianParams) -> Callable[[float, float], float]:
    """
    Returns a function v(x,y) for the double Gaussian potential.
    
    Formula: 
    V(x,y) = v0 - v0*[well_L + well_R] + v_b*[barrier] + delta*x
    
    This convention results in:
    - Potential at infinity ≈ v0
    - Potential at well minima ≈ 0
    - Potential at the top of the barrier ≈ v0 + v_b (at x=0)
    """
    sigma2 = p.sigma ** 2
    sigma_b2 = p.sigma_b ** 2

    def v(x: float, y: float) -> float:
        # Distance squared from left and right dot centers
        r_left2 = (x + p.a)**2 + y**2
        r_right2 = (x - p.a)**2 + y**2
        
        # Attractive potential from the two wells (depth v0)
        wells = p.v0 * (math.exp(-r_left2 / sigma2) + math.exp(-r_right2 / sigma2))
        
        # Repulsive central barrier (height v_b)
        barrier = p.v_b * math.exp(-(x**2) / sigma_b2)
        
        # Total potential: baseline - wells + barrier + detuning
        # The baseline (v_off) is set to v0 to shift the minima to zero.
        return p.v0 - wells + barrier + p.delta * x

    return v

def from_dimensionless_targets(
    a: float,
    well_depth: float,
    well_width: float,
    barrier_height: float,
    barrier_width: float,
    delta: float = 0.0,
) -> GaussianParams:
    """
    Constructs Gaussian parameters from dimensionless target values.
    
    Args:
        a: Half-separation of the wells.
        well_depth: The depth of the wells (v0). This also sets the potential at infinity.
        well_width: The width sigma of the wells.
        barrier_height: The height of the central barrier *above* the baseline at infinity.
        barrier_width: The width sigma_b of the barrier.
        delta: Detuning parameter to tilt the potential.
    
    Returns:
        A GaussianParams object with the specified parameters.
    """
    return GaussianParams(
        a=a,
        v0=well_depth,
        sigma=well_width,
        v_b=barrier_height,
        sigma_b=barrier_width,
        delta=delta
    )

def suggest_default() -> tuple[GaussianParams, GaAs]:
    """

    Suggests a default set of numerically stable parameters for a
    typical GaAs double quantum dot.
    
    Returns:
        A tuple containing the default GaussianParams and GaAs material properties.
    """
    mat = GaAs()
    
    # These parameters create a well-behaved potential for stable training.
    params = from_dimensionless_targets(
        a=1.5,
        well_depth=10.0,
        well_width=0.7,  # A "softer" width for stability
        barrier_height=4.0,
        barrier_width=0.3,
        delta=0.0
    )
    
    return params, mat

def potential_info(p: GaussianParams, mat: GaAs | None = None) -> str:
    """
    Returns a human-readable summary of the potential in physical units.
    """
    mat = mat or GaAs()
    
    # Convert dimensionless parameters to physical units
    sep_nm = 2 * p.a * mat.L0_nm
    depth_meV = p.v0 * mat.E0_meV
    barrier_meV = p.v_b * mat.E0_meV
    
    info = "Gaussian Double Quantum Dot (All-Positive Potential):\n"
    info += f"  Physical Separation: {sep_nm:.1f} nm\n"
    info += f"  Well Depth / Baseline V(inf): {depth_meV:.2f} meV\n"
    info += f"  Barrier Height (above baseline): {barrier_meV:.2f} meV\n"
    info += f"  Dimensionless well width (sigma): {p.sigma:.3f}"
    
    return info