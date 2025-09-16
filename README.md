# PINN for Double Quantum Dot (DQD) Eigenstates in GaAs

This repository implements a specialized Physics-Informed Neural Network (PINN) approach for solving the 2D Schrödinger eigenvalue problem in double quantum dots. Our method combines **SIREN networks** with a **hybrid Rayleigh-Ritz/PDE-residual formulation** to efficiently learn oscillatory quantum eigenstates without traditional meshing or matrix diagonalization.

## Key Innovation: Why This Approach Works

**SIREN for Quantum Mechanics**: We use Sinusoidal Representation Networks (SIREN) because quantum wavefunctions are inherently oscillatory. Unlike standard MLPs that suffer from spectral bias (learning low frequencies first), SIREN's sinusoidal activations naturally capture high-frequency oscillations, interference patterns, and sharp nodal structures that characterize quantum eigenstates.

**Hybrid Loss Formulation**: Our approach combines:
- **Rayleigh-Ritz energy minimization** (variational principle) for robust eigenvalue approximation
- **PDE residual enforcement** for local Schrödinger equation satisfaction
- **Orthogonality constraints** for systematic excited state computation

This eliminates the need for curriculum learning or frequency ramping strategies required by conventional PINNs, enabling direct training on the full-frequency spectrum of quantum solutions.


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

**Goal**: Learn low-lying eigenstates (ground and first excited) and extract device-relevant parameters like tunnel coupling, energy splitting, and charge localization — all without traditional meshing or matrix diagonalization.

## Why This Approach Outperforms Alternatives

**vs. Traditional Finite Difference/Element Methods**:
- ✅ **No meshing**: Continuous representation adapts to any potential shape
- ✅ **No matrix storage**: Memory scales with network size, not grid resolution  
- ✅ **Smooth derivatives**: Analytical gradients via autograd, no numerical differentiation errors
- ✅ **Rapid iteration**: Change potential parameters without remeshing/reassembly

**vs. Standard PINN Approaches**:
- ✅ **Direct high-frequency learning**: SIREN eliminates spectral bias without curriculum
- ✅ **Robust eigenvalue computation**: Rayleigh-Ritz provides variational bound, not just PDE satisfaction
- ✅ **Systematic excited states**: Orthogonality constraints prevent mode collapse
- ✅ **Physical insight**: Energy-based training connects to quantum mechanical principles

**vs. Curriculum/Frequency-Ramping PINNs**:
- ✅ **No frequency scheduling**: SIREN learns all scales simultaneously
- ✅ **Fewer hyperparameters**: No need to tune ramping schedules or frequency bands
- ✅ **Faster convergence**: Direct optimization without multi-stage training


## 2) Our PINN Architecture and Training Strategy

### Why SIREN Networks Excel for Quantum Problems

**Spectral Bias Solution**: Traditional neural networks learn low frequencies first, requiring careful curriculum learning for oscillatory PDEs. SIREN's sinusoidal activations `sin(w₀x)` naturally represent the Fourier components of quantum wavefunctions, enabling direct learning of:
- Rapid oscillations in kinetic-dominated regions
- Sharp interference fringes at potential barriers  
- Precise nodal structures in excited states
- Smooth exponential decay in classically forbidden regions

**Accurate Higher-Order Derivatives**: Quantum mechanics requires precise second derivatives (Laplacian) for kinetic energy. SIREN's smooth periodic activations provide stable, accurate ∇²ψ computation via automatic differentiation, crucial for eigenvalue accuracy.

### Network Architecture
- **Model**: SIREN with frequency-scaled sine activations
- **Input**: 2D coordinates (x, y) in dimensionless units
- **Output**: Real-valued wavefunction ψ(x, y)
- **Architecture**: Configurable depth/width (default: 6 layers × 128 neurons)
- **Precision**: Double precision (float64) for derivative stability

### Hybrid Physics-Informed Loss Strategy

Our approach combines **global energy minimization** with **local PDE enforcement** for robust eigenstate learning:

#### 1. **Rayleigh-Ritz Energy** (Primary Objective)
```
E_RR(ψ) = [∫ (|∇ψ|² + v ψ²) dΩ] / [∫ ψ² dΩ]
```
- **Variational principle**: Minimizing energy drives toward true eigenstates
- **Scale-invariant**: Ratio form handles normalization automatically  
- **Gradient-squared kinetic**: Avoids boundary terms, works with soft walls
- **Global accuracy**: Computed on structured quadrature grid (nq × nq)

#### 2. **PDE Residual** (Local Constraint)
```
L_residual = ⟨|Hψ - E_local ψ|²⟩_collocation
```
- **Schrödinger enforcement**: Ensures ψ satisfies eigenvalue equation locally
- **Adaptive sampling**: Random collocation points (nc per epoch) capture fine details
- **Projected residual**: Removes local energy to prevent trivial solutions

#### 3. **Excited State Handling** (Deflation Method)
```
L_ortho = [⟨ψ_new, ψ_ref⟩ / (||ψ_new|| ||ψ_ref||)]²
```
- **Systematic orthogonality**: Enforces ⟨ψ_n|ψ_m⟩ = 0 for n ≠ m
- **Frozen reference**: Ground state model provides deflation constraint
- **No mode collapse**: Prevents excited states from converging to ground state

#### 4. **Optional Constraints**
- **Normalization**: L_norm = (∫ ψ² dΩ - 1)² maintains unit norm
- **Symmetry**: Parity constraints for symmetric potentials (δ = 0)

**Total Loss**: L = λ_rr·E_RR + λ_pde·L_residual + λ_norm·L_norm + [λ_ortho·L_ortho] + [λ_sym·L_sym]

### Training Strategy

**Two-Stage Optimization**:
1. **Adam phase**: First-order optimization for broad convergence (1000+ epochs)
2. **L-BFGS refinement**: Second-order polish for sharp minima (200 iterations)

**Key Advantages of Our Approach**:
- **No curriculum needed**: SIREN handles full frequency spectrum from start
- **Stable derivatives**: Double precision + smooth activations ensure accurate ∇²ψ
- **Soft boundaries**: Potential confinement eliminates boundary condition complexity
- **Systematic excited states**: Orthogonality constraints enable reliable higher eigenstate computation

**Computational Efficiency**:
- **Mesh-free**: No spatial discretization or matrix assembly
- **Scalable**: O(N) complexity in network parameters, not grid points
- **GPU-friendly**: Fully differentiable, vectorized operations


## 3) Quantum Mechanical Results and Physical Insights

### Eigenstate Characteristics
- **Ground state (bonding)**: Two lobes at (±a, 0) with even parity, no nodal line
- **First excited (antibonding)**: Nodal line at x=0, odd parity, higher energy
- **Energy splitting**: ΔE = E₁ - E₀ ≈ 2t (tunnel coupling) for symmetric dots
- **Detuning effects**: δ ≠ 0 creates avoided crossings, charge transfer

### Training Diagnostics
- **Energy convergence**: Variational bound ensures E ≥ E_exact
- **PDE residual**: Target 10⁻² to 10⁻¹ (balance accuracy vs. computational cost)
- **Normalization**: ∫|ψ|² ≈ 1 (quantum probability conservation)
- **Orthogonality**: ⟨ψ₀|ψ₁⟩ ≈ 0 for excited states (quantum mechanical requirement)

### Physical Observables (Automated Analysis)
Our framework automatically extracts device-relevant parameters:
- **Tunnel coupling**: t ≈ (E₁ - E₀)/2 for symmetric case
- **Charge localization**: Left/right dot populations from ∫|ψ|² over regions
- **Density peaks**: Wavefunction maxima near potential minima (±a, 0)
- **Center of mass**: ⟨x⟩, ⟨y⟩ for charge distribution analysis
- **Energy scales**: Dimensionless → physical units via E₀ ≈ 0.63 meV (GaAs, L₀=30nm)

### Validation Metrics
- **Eigenvalue accuracy**: Relative error vs. finite difference baselines
- **Wavefunction fidelity**: Overlap with reference solutions
- **Physical consistency**: Energy ordering, symmetry properties, normalization


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

#### Parity options (symmetric case, δ = 0)
- To steer GS toward even parity: add `--lam-sym-even 0.1` (typically small, 0.05–0.3)
- To steer ES toward odd parity: add `--lam-sym-odd 0.1` together with `--lam-ortho` and a GS reference

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
- For symmetric double wells (δ ≈ 0), small parity weights help target the expected eigenmodes:
  - GS even: `--lam-sym-even 0.05–0.3`
  - ES odd:  `--lam-sym-odd 0.05–0.3` (use with `--lam-ortho` and a GS `--ref-model`)



