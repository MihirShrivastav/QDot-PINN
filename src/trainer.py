# train_gaussian_dqd.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataclasses import dataclass

# --- Assumes a project structure where these modules are accessible ---
from potentials.gaussian import GaussianParams, from_dimensionless_targets, v_gaussian_factory, potential_info
from physics.units import GaAs, pretty_summary
from pinn.models import SIREN
from pinn.losses import rayleigh_ritz_energy, pde_residual_loss, orthogonality_loss
from viz.plotting import plot_potential, plot_wavefunction_density, plot_training_curves

# Use double precision for more accurate second derivatives
torch.set_default_dtype(torch.float64)

# --- 1. Central Configuration Block ---

@dataclass
class Config:
    """A single place to configure the simulation."""
    # Potential Parameters (Dimensionless)
    a: float = 1.5           # Half-separation of wells
    v0: float = 10.0         # Well depth and potential at infinity
    sigma: float = 0.7       # Well width (a "soft" value for stability)
    v_b: float = 4.0         # Barrier height above the baseline
    sigma_b: float = 0.3     # Barrier width
    delta: float = 0.0       # Detuning (0.0 for symmetric wells)

    # PINN Model Architecture
    hidden_features: int = 128
    hidden_layers: int = 6

    # Training Hyperparameters
    epochs_adam: int = 2000     # Number of epochs for the Adam optimizer
    epochs_lbfgs: int = 400     # Max iterations for the L-BFGS optimizer
    learning_rate: float = 5e-3  # Learning rate for Adam

    # Loss Function Weights (standard, no curriculum)
    lam_pde: float = 2.5
    lam_rr: float = 1.0
    # Normalization weight is kept high
    lam_norm: float = 100.0

    lam_bc: float = 200.0  # Strong penalty for non-zero boundary values
    # Orthogonality (optional; for excited states)
    lam_ortho: float = 0.0
    ref_model_path: str | None = None


    # Simulation Domain and Resolution
    domain_size: float = 5.0     # Simulation box will be [-size, size]
    nq: int = 128                # Resolution of quadrature grid for loss calculation
    nc: int = 4096               # Number of collocation points per epoch

    # Output
    output_dir: str = "data/gaussian_gs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions ---

def vectorized_potential(p: GaussianParams, xy: torch.Tensor) -> torch.Tensor:
    """Vectorized potential evaluation using PyTorch for batch processing."""
    x, y = xy[:, 0:1], xy[:, 1:2]

    # Note: v_gaussian_factory from your file uses math.exp, which doesn't work on tensors.
    # We must use torch.exp. A torch-native version is required.
    sigma2, sigma_b2 = p.sigma**2, p.sigma_b**2
    r_left2 = (x + p.a)**2 + y**2
    r_right2 = (x - p.a)**2 + y**2
    wells = p.v0 * (torch.exp(-r_left2 / sigma2) + torch.exp(-r_right2 / sigma2))
    barrier = p.v_b * torch.exp(-(x**2) / sigma_b2)
    return p.v0 - wells + barrier + p.delta * x

def make_quadrature_grid(size: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Creates a grid of points and corresponding weights for numerical integration."""
    xs = torch.linspace(-size, size, n, device=device)
    ys = torch.linspace(-size, size, n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([Xg.flatten(), Yg.flatten()], dim=-1)
    cell_area = ((2 * size) / (n - 1))**2
    weights = torch.full((xy.shape[0], 1), cell_area, device=device)
    return xy, weights

def normalization_penalty(model: torch.nn.Module, xy_q: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    """Calculates the normalization penalty: (∫|ψ|² dΩ - 1)²."""
    psi = model(xy_q)
    norm = torch.sum(w_q * psi**2)
    return (norm - 1.0)**2

def sample_boundary(size: float, n: int, device: str) -> torch.Tensor:
    """Samples points from the four edges of the square domain."""
    n_edge = n // 4
    # Top and Bottom edges
    x_tb = (torch.rand(2 * n_edge, 1, device=device) * 2 - 1) * size
    y_tb = torch.cat([torch.full((n_edge, 1), size, device=device), 
                      torch.full((n_edge, 1), -size, device=device)], dim=0)
    tb_pts = torch.cat([x_tb, y_tb], dim=1)
    
    # Left and Right edges
    y_lr = (torch.rand(2 * n_edge, 1, device=device) * 2 - 1) * size
    x_lr = torch.cat([torch.full((n_edge, 1), -size, device=device), 
                      torch.full((n_edge, 1), size, device=device)], dim=0)
    lr_pts = torch.cat([x_lr, y_lr], dim=1)
    
    return torch.cat([tb_pts, lr_pts], dim=0)

# --- Main Script ---

if __name__ == "__main__":
    # 1. Setup Configuration and Environment
    cfg = Config()
    device = torch.device(cfg.device)
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # 2. Define the Physical System
    mat = GaAs()
    params = from_dimensionless_targets(
        a=cfg.a, well_depth=cfg.v0, well_width=cfg.sigma,
        barrier_height=cfg.v_b, barrier_width=cfg.sigma_b, delta=cfg.delta
    )
    print("--- System Configuration ---")
    print(pretty_summary(mat))
    print(potential_info(params, mat))
    print("-" * 28)

    # 3. Create Model, Grids, and Optimizer
    model = SIREN(in_features=2, hidden_features=cfg.hidden_features,
                  hidden_layers=cfg.hidden_layers, out_features=1).to(device)
    # Optional reference model for orthogonality (for excited states)
    ref_model = None
    if cfg.lam_ortho > 0.0 and cfg.ref_model_path:
        ref_path = Path(cfg.ref_model_path)
        if ref_path.exists():
            ref_model = SIREN(in_features=2, hidden_features=cfg.hidden_features,
                              hidden_layers=cfg.hidden_layers, out_features=1).to(device)
            ref_model.load_state_dict(torch.load(ref_path, map_location=device))
            ref_model.eval()
            for p_ref in ref_model.parameters():
                p_ref.requires_grad_(False)
            print(f"Loaded reference model for orthogonality from: {ref_path}")
        else:
            print(f"Warning: lam_ortho>0 but ref_model_path not found: {ref_path}")
    xy_q, w_q = make_quadrature_grid(cfg.domain_size, cfg.nq, device)

    vfun_batch = lambda xy: vectorized_potential(params, xy)

    # 4. Training: Adam then L-BFGS (no curriculum)
    print(f"\n--- Starting Training on device: {device} ---")

    # == STAGE 1: Adam Optimizer to find the general shape ==
    print("\n--- STAGE 1: Finding Wavefunction Shape (Adam) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history = {"E": [], "Lres": [], "Lnorm": []}

    for epoch in range(1, cfg.epochs_adam + 1):
        model.train()
        xy_c = (torch.rand((cfg.nc, 2), device=device) * 2 - 1) * cfg.domain_size

        E = rayleigh_ritz_energy(model, vfun_batch, xy_q, w_q)
        Lres = pde_residual_loss(model, vfun_batch, xy_c)
        Lnorm = normalization_penalty(model, xy_q, w_q)


        loss = cfg.lam_rr * E + cfg.lam_pde * Lres + cfg.lam_norm * Lnorm

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history["E"].append(E.item())
        history["Lres"].append(Lres.item())
        history["Lnorm"].append(Lnorm.item())

        if epoch % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch:5d}] E={E.item():.6f} Lres={Lres.item():.3e} Lnorm={Lnorm.item():.3e}")

    # == STAGE 2: L-BFGS Optimizer to find the precise minimum energy ==
    print("\n--- STAGE 2: Minimizing Energy (L-BFGS) ---")
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=cfg.epochs_lbfgs, line_search_fn="strong_wolfe")
    xy_c_lbfgs = (torch.rand((cfg.nc, 2), device=device) * 2 - 1) * cfg.domain_size


    def closure():
        optimizer.zero_grad(set_to_none=True)
        E = rayleigh_ritz_energy(model, vfun_batch, xy_q, w_q)
        Lres = pde_residual_loss(model, vfun_batch, xy_c_lbfgs)
        Lnorm = normalization_penalty(model, xy_q, w_q)
        loss = cfg.lam_rr * E + cfg.lam_pde * Lres + cfg.lam_norm * Lnorm
        loss.backward()
        return loss

    optimizer.step(closure)

    # 5. Final Evaluation and Visualization
    print("\n--- Training Complete: Evaluating and Plotting ---")
    model.eval()
    final_E = rayleigh_ritz_energy(model, vfun_batch, xy_q, w_q).item()
    print(f"Final Ground State Energy E_0 = {final_E:.6f} (dimensionless)")
    print(f"Final Ground State Energy E_0 = {final_E * mat.E0_meV:.6f} (meV)")

    # Save artifacts
    torch.save(model.state_dict(), outdir / "model_final.pt")
    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Generate plots using your viz.plotting module

    # Plot 1: Training Curves
    plot_training_curves(history, out_path=str(outdir / "training_curves.png"))

    # Plot 2: Potential
    vfun_numpy = v_gaussian_factory(params)
    plot_potential(lambda x, y: vfun_numpy(float(x), float(y)), cfg.domain_size, cfg.domain_size, title="Potential V(x,y)")
    plt.savefig(outdir / "potential.png", dpi=200)

    # Create high-resolution grid for final wavefunction plots
    plot_res = 200
    xs = np.linspace(-cfg.domain_size, cfg.domain_size, plot_res)
    ys = np.linspace(-cfg.domain_size, cfg.domain_size, plot_res)
    xy_plot_torch = torch.stack(torch.meshgrid(torch.from_numpy(xs), torch.from_numpy(ys), indexing="xy"), dim=-1).reshape(-1, 2).to(device)

    with torch.no_grad():
        psi_vec = model(xy_plot_torch).cpu().numpy().flatten()

    # Plot 3: 2D Probability Density |ψ(x,y)|²
    plot_wavefunction_density(
        xs, ys, psi_vec, plot_res, plot_res,
        title=r"Final Probability Density $|\psi|^2$",
        overlay_points=[(-params.a, 0.0), (params.a, 0.0)]
    )
    plt.savefig(outdir / "final_psi_density_2D.png", dpi=300)

    # Plot 4 (NEW): 1D Slice of Potential and Energy
    v_slice = vectorized_potential(params, torch.stack([torch.from_numpy(xs), torch.zeros_like(torch.from_numpy(xs))], dim=-1).to(device)).cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(xs, v_slice, 'k-', label="Potential V(x,0)")
    plt.axhline(y=final_E, color='r', linestyle='--', label=f"Ground State Energy E_0 = {final_E:.3f}")
    plt.title("Potential Profile and Ground State Energy")
    plt.xlabel("x (dimensionless)")
    plt.ylabel("Energy (dimensionless)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(outdir / "final_energy_level.png", dpi=200)

    # Plot 5 (NEW): 1D Slice of Wavefunction ψ(x,0)
    psi_slice = psi_vec.reshape(plot_res, plot_res)[:, plot_res // 2]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, psi_slice, 'b-', label="Wavefunction ψ(x,0)")
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.title("Wavefunction Profile (Slice at y=0)")
    plt.xlabel("x (dimensionless)")
    plt.ylabel("ψ(x, 0)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(outdir / "final_psi_slice.png", dpi=200)

    print(f"\n✅ All results and plots saved to '{outdir}' directory.")