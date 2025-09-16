# Training Parameters Reference

This document explains all command-line parameters for the `src/train_1e.py` script used to train Physics-Informed Neural Networks for single-electron quantum dot eigenstates.

## Potential Parameters

The training script supports two potential types via `--potential` flag:

### Common Parameters (Both Potential Types)

### `--a` (float, default: 1.5)
**Half-separation between quantum dots**
- Controls the distance between the two potential minima at (±a, 0)
- In dimensionless units (multiply by L₀ = 30 nm for physical distance)
- Larger values → wider separation → weaker tunnel coupling
- Typical range: 1.2 - 1.8

### `--c4` (float, default: None)
**Quartic coefficient for x-direction confinement**
- Controls the barrier height and curvature around the potential minima
- If not provided, calculated from `--hbar-omega-x` target
- Larger values → steeper wells → stronger confinement
- Typical range: 0.5 - 2.0

### `--c2y` (float, default: None)
**Quadratic coefficient for y-direction confinement**
- Controls transverse (y-direction) harmonic confinement
- If not provided, calculated from `--hbar-omega-y` target
- Larger values → tighter y-confinement
- Typical range: 10 - 20

### `--delta` (float, default: 0.0)
**Linear detuning term**
- Creates energy difference between left and right dots
- δ = 0: symmetric double dot
- δ > 0: right dot has higher energy
- δ < 0: left dot has higher energy
- Controls avoided crossing behavior in energy sweeps

### `--hbar-omega-x` (float, default: 3.0)
**Target harmonic frequency in x-direction (meV)**
- Used to calculate `c4` if not explicitly provided
- Represents the desired oscillation frequency near potential minima
- Physical units: meV
- Typical range: 2.0 - 5.0 meV

### `--hbar-omega-y` (float, default: 5.0)
**Target harmonic frequency in y-direction (meV)**
- Used to calculate `c2y` if not explicitly provided
- Controls transverse confinement strength
- Physical units: meV
- Typical range: 3.0 - 8.0 meV

## Gaussian Potential Parameters

For `--potential gaussian`, the potential form is:
`v(x,y) = v_off - v₀[exp(-((x-a)² + y²)/σ²) + exp(-((x+a)² + y²)/σ²)] + v_b·exp(-x²/σ_b²) + δ·x`

### `--v0` (float, default: 10.0)
**Well depth (dimensionless)**
- Depth of the Gaussian wells (attractive potential)
- Larger values → deeper wells → stronger confinement
- Typical range: 6.0 - 15.0
- Controls single-particle energy levels

### `--sigma` (float, default: 0.4)
**Well width parameter (dimensionless)**
- Controls spatial extent of Gaussian wells
- Smaller values → tighter confinement → higher energy levels
- Typical range: 0.3 - 0.6
- Should be smaller than separation `a` to avoid well overlap

### `--v-b` (float, default: 4.0)
**Barrier height (dimensionless)**
- Height of central Gaussian barrier between wells
- Controls tunnel coupling exponentially
- Larger values → higher barrier → weaker coupling
- Typical range: 0.0 - 10.0

### `--sigma-b` (float, default: 0.3)
**Barrier width parameter (dimensionless)**
- Controls spatial extent of central barrier
- Smaller values → narrower barrier → stronger coupling
- Typical range: 0.2 - 0.5
- Should be comparable to well separation for realistic barriers

## Neural Network Architecture

### `--hidden` (int, default: 128)
**Number of neurons in each hidden layer**
- Controls model capacity and expressiveness
- More neurons → better approximation but slower training
- Typical range: 64 - 256
- Must balance accuracy with computational cost

### `--layers` (int, default: 6)
**Number of hidden layers in SIREN network**
- Deeper networks can capture more complex wavefunction features
- Too deep may cause training instability
- Typical range: 4 - 8
- SIREN architecture handles depth better than standard MLPs

## Training Hyperparameters

### `--epochs` (int, default: 1000)
**Number of Adam optimization epochs**
- Total training iterations before optional L-BFGS refinement
- Monitor convergence in training curves to adjust
- Typical range: 500 - 3000
- More epochs needed for complex potentials or excited states

### `--lr` (float, default: 1e-3)
**Learning rate for Adam optimizer**
- Controls step size in gradient descent
- Too high → unstable training
- Too low → slow convergence
- Typical range: 1e-4 to 1e-2
- May need reduction for excited states

### `--lbfgs-iters` (int, default: 200)
**L-BFGS refinement iterations**
- Second-order optimization after Adam training
- Set to 0 to disable L-BFGS refinement
- Helps achieve sharper minima and better convergence
- Typical range: 100 - 500

## Numerical Integration and Sampling

### `--nq` (int, default: 96)
**Quadrature grid points per axis**
- Creates nq × nq grid for numerical integration
- Used for Rayleigh-Ritz energy and normalization integrals
- Higher values → more accurate integration but slower
- Total grid points: nq²
- Typical range: 64 - 128

### `--nc` (int, default: 4096)
**Collocation points per epoch**
- Random points sampled each epoch for PDE residual evaluation
- More points → better PDE satisfaction but slower training
- Uniform sampling over the computational domain
- Typical range: 2048 - 8192

## Loss Function Weights

### `--lam-rr` (float, default: 1.0)
**Rayleigh-Ritz energy weight**
- Controls importance of energy minimization
- Primary physics-based objective
- Usually kept at 1.0 as reference scale
- Adjust other weights relative to this

### `--lam-pde` (float, default: 1.0)
**PDE residual loss weight**
- Enforces Schrödinger equation satisfaction
- Higher values → stricter PDE compliance
- Balance with energy minimization
- Typical range: 0.5 - 2.0

### `--lam-norm` (float, default: 1.0)
**Normalization penalty weight**
- Enforces ∫|ψ|² dΩ ≈ 1
- Prevents trivial ψ = 0 solution
- Increase if normalization drifts during training
- Typical range: 1.0 - 100.0

## Excited State Parameters

### `--state` (int, default: 0)
**Eigenstate index to compute**
- 0: Ground state (lowest energy)
- 1: First excited state (requires reference model)
- Higher values not currently supported
- Excited states need orthogonality constraints

### `--ref-model` (str, default: None)
**Path to reference ground state model**
- Required for excited state training (state > 0)
- Should point to `model_best.pt` from ground state run
- Used to enforce orthogonality via overlap penalty
- Example: `data/run_1e_gs/model_best.pt`

### `--lam-ortho` (float, default: 10.0)
**Orthogonality penalty weight**
- Enforces orthogonality to reference state
- Only used when `--ref-model` is provided
- Higher values → stricter orthogonality
- Typical range: 5.0 - 50.0

## Symmetry Constraints

### `--lam-sym-even` (float, default: 0.0)
**Even parity penalty weight**
- Encourages ψ(x,y) ≈ ψ(-x,y) (symmetric about x=0)
- Useful for ground state when δ = 0
- Small values recommended: 0.05 - 0.3
- Set to 0.0 to disable

### `--lam-sym-odd` (float, default: 0.0)
**Odd parity penalty weight**
- Encourages ψ(x,y) ≈ -ψ(-x,y) (antisymmetric about x=0)
- Useful for first excited state when δ = 0
- Small values recommended: 0.05 - 0.3
- Set to 0.0 to disable

## System and Output Parameters

### `--device` (str, default: "cpu")
**Computation device**
- "cpu": Use CPU computation
- "cuda": Use GPU acceleration (if available)
- GPU recommended for large networks or high resolution
- Check GPU memory for large `--nq` or `--nc` values

### `--outdir` (str, default: "data/run_1e")
**Output directory path**
- Where all results are saved
- Creates directory if it doesn't exist
- Contains: model weights, plots, metrics, configuration
- Use descriptive names: `data/run_1e_gs`, `data/run_1e_es_01`

### `--grid-save` (int, default: 200)
**Evaluation grid resolution**
- Creates N×N grid for wavefunction visualization
- Higher values → smoother plots but larger files
- Used for density analysis and plotting
- Typical range: 100 - 400

### `--no-plots` (flag, default: False)
**Disable plot generation**
- Skip creating PNG visualization files
- Useful for batch runs or when only metrics are needed
- Still saves numerical data (energies, history, etc.)

## Visualization Mode

### `--viz-only` (flag, default: False)
**Visualization-only mode**
- Load existing model and regenerate outputs without training
- Useful for creating plots with different parameters
- Requires existing model weights in output directory

### `--model` (str, default: None)
**Custom model path for viz-only mode**
- Override default model path in viz-only mode
- Default: `<outdir>/model_best.pt`
- Allows loading models from different locations

## Usage Examples

### Basic ground state training:
```bash
python -m src.train_1e --outdir data/gs_run --epochs 1500
```

### Excited state with orthogonality:
```bash
python -m src.train_1e --state 1 --ref-model data/gs_run/model_best.pt \
  --outdir data/es_run --lam-ortho 20.0
```

### High-resolution GPU training:
```bash
python -m src.train_1e --device cuda --nq 128 --nc 8192 \
  --hidden 256 --layers 8
```

### Symmetric case with parity constraints:
```bash
python -m src.train_1e --delta 0.0 --lam-sym-even 0.1 \
  --outdir data/symmetric_gs
```

### Visualization only:
```bash
python -m src.train_1e --viz-only --outdir data/gs_run \
  --grid-save 300
```

## Gaussian Potential Examples

### Basic Gaussian ground state:
```bash
python -m src.train_flexible --potential gaussian \
  --a 1.5 --v0 8.0 --sigma 0.35 --v-b 3.0 --sigma-b 0.25 \
  --outdir data/gaussian_gs
```

### Gaussian excited state:
```bash
python -m src.train_flexible --potential gaussian --state 1 \
  --ref-model data/gaussian_gs/model_best.pt \
  --outdir data/gaussian_es --lam-ortho 15.0
```

### Strong coupling (low barrier):
```bash
python -m src.train_flexible --potential gaussian \
  --v-b 1.0 --sigma-b 0.4 --outdir data/strong_coupling
```

### Weak coupling (high barrier):
```bash
python -m src.train_flexible --potential gaussian \
  --v-b 8.0 --sigma-b 0.2 --outdir data/weak_coupling
```

### Detuned system:
```bash
python -m src.train_flexible --potential gaussian \
  --delta 2.0 --outdir data/detuned_gaussian
```