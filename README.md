# PINN for Double Quantum Dot (DQD) Eigenstates in GaAs

This repository trains a Physics‑Informed Neural Network (PINN) to approximate single‑electron stationary states (ground and first excited) in a 2D double quantum dot potential relevant to GaAs devices. It emphasizes: a clear physical model, a mesh‑free PINN solution, informative visualizations/metrics, and reproducible runs.


## 1) Physical problem

- We solve the time‑independent Schrödinger eigenproblem for a single electron in 2D:
  - Dimensionless form: find (ψ, E) such that
    -Hψ + v(x, y) ψ = E ψ on a finite box Ω = [−X, X] × [−Y, Y]
  - We use “soft walls” (no hard boundary conditions); the confining potential grows outside the wells to localize ψ within Ω.
- Potential (biquadratic double well; implemented):
  - v(x, y) = c4 (x² − a²)² + c2y y² + δ x
    - a controls well separation
    - c4, c2y set the x and y curvatures (via target harmonic energies if desired)
    - δ is an optional detuning along x
- Materials and non‑dimensionalization (GaAs defaults):
  - Characteristic length L0 = 30 nm
  - Energy scale E0 ≈ 0.6318 meV
  - Dimensionless equations use x′ = x/L0, E′ = E/E0, etc. A handy summary is printed at run start.

Goal: learn the low‑lying eigenstates — ground (bonding) and first excited (antibonding) — and estimate physically meaningful quantities like normalization, energy, density peaks, mass on left/right dots, and values near the potential minima (±a, 0).


## 2) How the PINN solves it

This PINN learns ψ(x, y) directly, avoiding spatial meshes and eigen‑solves of large matrices. Training balances multiple physically‑motivated loss terms.

### Network
- Model: SIREN (sinusoidal representation network) with sin activations to capture oscillatory solutions.
- Architecture (configurable): input 2D → [hidden × layers] → output 1D (ψ).

### Quadrature and collocation
- Quadrature grid (size `nq × nq`) for integrals (Rayleigh quotient, normalization, overlaps). Uniform weights.
- Random collocation points (`nc` per epoch) uniformly sampled in the box for PDE residual.

### Loss components (per epoch)
1. Rayleigh–Ritz energy (scale‑invariant quotient)
   - E_RQ(ψ) = [∫ (|∇ψ|² + v ψ²) dΩ] / [∫ ψ² dΩ]
   - Uses gradient‑squared kinetic energy; stable and positive‑definite with soft walls.
2. PDE residual (eigen‑equation consistency)
   - Compute Hψ = −∇²ψ + vψ, project out the local Rayleigh estimate e_local, and penalize the mean squared residual of Hψ − e_local ψ at collocation points.
3. Normalization penalty (optional but helpful)
   - L_norm = (∫ ψ² dΩ − 1)², encouraging ∫ ψ² ≈ 1 on the quadrature grid.
4. Orthogonality (excited states)
   - L_ortho = [⟨ψ_new, ψ_ref⟩ / (||ψ_new||·||ψ_ref||)]² on the quadrature grid (squared, so lower is better). ψ_ref is a frozen ground‑state model.

Total objective:  L = λ_rr·E_RQ + λ_pde·L_res + λ_norm·L_norm [+ λ_ortho·L_ortho]

### Optimization
- Adam (first‑order) for many epochs.
- Optional LBFGS refinement (second‑order) at the end for a sharper minimum.
- Gradients computed via PyTorch autograd; we take care to avoid common LBFGS/closure pitfalls.

### Practical details
- Soft boundaries: the box is finite, but we do not impose Dirichlet/Neumann BCs; the potential confines ψ.
- Visualization normalization: for plots/metrics we normalize ψ on the plotting grid so ∫|ψ|² ≈ 1, ensuring interpretable density heatmaps and CSV metrics even when raw amplitude varies.
- Orthogonality: excited‑state runs require a reference ground‑state model to enforce deflation via overlap penalty.


## 3) Expected results and diagnostics

- Ground state: two lobes centered near (x, y) ≈ (±a, 0) with bonding symmetry (no node at x = 0).
- First excited state: antibonding across the inter‑dot barrier — typically a node near x = 0, lobes again near (±a, 0).
- Normalization: ∫ ψ² dΩ ≈ 1 on the quadrature grid; reported as NormValue in history and as norm_value_final in energies.
- Residual: should decrease over training; useful ballpark is 1e−2 to 1e−1 depending on settings.
- Overlap0 (excited state): should be small (near 0) if orthogonality is working.
- Density features (CSV/JSON):
  - total_prob (from the plotting grid, should be ≈ 1),
  - center of mass (com_x, com_y ≈ 0 for symmetric case),
  - side masses (left/right),
  - top peaks (coordinates close to ±a along x and y ≈ 0),
  - minima_density values sampled at (±a, 0).

Energy scale: The reported E is dimensionless (divide by E0 to convert to meV). For typical GaAs‑like parameters, expect meV‑scale energies. The splitting between ground and first excited reflects the tunnel coupling (2t) for symmetric double wells; detuning δ shifts the relative dot energies.


## How to run

All commands should be run from the repository root.

### Ground state (state = 0)
```bash
python -m src.train_1e --outdir data/run_1e_gs --device cpu
```
Common options:
- Use CUDA: `--device cuda`
- Quadrature resolution: `--nq 96`
- Collocation per epoch: `--nc 4096`
- LBFGS refinement: `--lbfgs-iters 200` (set 0 to disable)
- Potential via targets: `--a 1.5 --hbar-omega-x 3.0 --hbar-omega-y 5.0 --delta 0.0`
- Or explicit coefficients: `--c4 <..> --c2y <..> --delta <..>`

### First excited state (orthogonal to ground; state = 1)
```bash
python -m src.train_1e --state 1 \
  --ref-model data/run_1e_gs/model_best.pt \
  --outdir data/run_1e_es \
  --lam-ortho 10.0 \
  --device cpu
```
You may carry over `--nq`, `--nc`, `--lbfgs-iters`, etc. from the ground‑state run. The reference model should be the best (frozen) ground‑state weights.

### Viz‑only mode (no training; re‑generate outputs from saved weights)
```bash
# Regenerate for a run directory (uses <outdir>/model_best.pt and <outdir>/config.json if present)
python -m src.train_1e --viz-only --outdir data/run_1e_gs

# Or specify a custom weights path
python -m src.train_1e --viz-only --outdir data/run_1e_gs --model path/to/model_best.pt
```
This writes `energies_eval.txt`, `psi_density.png`, `potential.png`, `density_features.json/csv`, and `psi_grid.pt` without retraining.


## Outputs and logs

Each run writes into `--outdir` (default `data/run_1e`):

- energies.txt
  - `E_final`, `residual_final`, `norm_value_final`, `norm_penalty_final`, `E_best`
- history.json / history.csv
  - Per‑epoch `E`, `Lres`, `Lnorm`, `NormValue` (∫ψ²), and `Overlap0` (if applicable)
- psi_density.png
  - Heatmap of |ψ|² with contour overlays and markers at (±a, 0); normalized on the plotting grid for clarity
- potential.png
  - Heatmap of v(x, y)
- density_features.json and density_features.csv
  - Quantitative summary: peaks, center of mass, side masses, minima densities, total_prob
- psi_grid.pt
  - Torch tensor with x, y, and psi on the plotting grid
- config.json
  - Captures key parameters (potential, box size, quadrature, model hyperparameters)
- model_best.pt
  - Best weights snapshot (lowest observed E during training)

Viz‑only mode writes a separate `energies_eval.txt` to avoid clobbering the original `energies.txt`.


## Tips and troubleshooting

- If NormValue drifts far from 1 during training, increase `--lam-norm` (e.g., 50–200) or rely more on the scale‑invariant Rayleigh quotient by down‑weighting `--lam-norm` later in training.
- If the excited state looks too similar to the ground state, raise `--lam-ortho` moderately (e.g., 10→50), and ensure you point to the correct ground‑state `--ref-model`.
- Higher `--nq` (quadrature) and `--nc` (collocation) improve accuracy but increase cost.
- Use `--lbfgs-iters 0` to skip LBFGS for faster iteration; re‑enable later for refinement.
- The plotting grid normalization makes density plots comparable across runs even if raw amplitude varies.


## Acknowledgments

- PINN approach inspired by the Rayleigh–Ritz principle and physics‑informed residual minimization.
- SIREN networks (sine activations) help represent oscillatory quantum states efficiently.

