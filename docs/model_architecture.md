# Model Architecture: SIREN Network for ψ(x, y) Approximation

This document explains the SIREN (Sinusoidal Representation Network) architecture implemented in `src/pinn/models.py` that approximates the real-valued wavefunction ψ(x, y) for the 2D, single-electron Schrödinger eigenproblem.

The model takes 2D inputs (x, y) and outputs a single scalar ψ. It is differentiable end-to-end; we compute ∇ψ, ∇²ψ via autograd for the physics-informed loss functions.


## SIREN Architecture

SIREN is well-suited to representing oscillatory solutions and sharp interference patterns that occur in quantum mechanical eigenstates. It uses sine activations with frequency scaling.

- Module: `SIREN`
- Layers: `SIRENLayer` blocks (Linear → Sine) followed by a final Linear to 1D output
- Constructor signature:
  - `SIREN(in_features: int, hidden_features: int = 128, hidden_layers: int = 4, out_features: int = 1, w0: float = 30.0, w0_hidden: float = 1.0)`
- Activation:
  - First layer uses `Sine(w0)` (default `w0 = 30.0`) to quickly excite a broad range of spatial frequencies from raw coordinates
  - Hidden layers use `Sine(w0_hidden)` (default `w0_hidden = 1.0`) for stability
- Initialization (from code):
  - First layer weights: `Uniform(-1/in_features, 1/in_features)`
  - Hidden layer weights: `Uniform(-sqrt(6/in_features)/w0_hidden, +sqrt(6/in_features)/w0_hidden)`
  - Biases are zeroed

Why SIREN works well here
- Quantum eigenfunctions are oscillatory; sinusoidal activations directly model such structure
- We rely on accurate second derivatives (for ∇²ψ). SIREN’s smooth periodic activations yield stable higher-order derivatives
- Combined with double precision (we set `torch.set_default_dtype(torch.float64)`), this improves Laplacian accuracy

Parameter count (approx.)
- For `in=2, hidden=128, layers=4, out=1`:
  - First: (2×128 + 128) = 384
  - Hidden (3 blocks): 3 × (128×128 + 128) = 3 × 16512 = 49536
  - Output: (128×1 + 1) = 129
  - Total ≈ 50049 parameters

Tips
- If you increase `w0` too much, training may become unstable; if too small, you might underfit high-frequency structure
- Keep coordinates roughly O(1) (our domain construction does this) to avoid extreme w0·x values
- Start with `hidden=128, layers=4` and scale up if residuals/energies plateau prematurely





## Shapes, dtypes, and derivatives

- Inputs: `[N, 2]` float tensor with coordinates (x, y)
- Outputs: `[N, 1]` scalar ψ values
- Dtype: we set default to float64 (double). This improves stability for second derivatives
- Derivatives: we rely on autograd for ∇ψ and ∇²ψ (see `src/pinn/losses.py`). When computing derivatives, make sure `xy.requires_grad_(True)` is set as done in the loss functions


## Architecture Configuration

### Recommended defaults
- Use SIREN with: `hidden_features=128, hidden_layers=6, w0=30.0, w0_hidden=1.0`
- If training is noisy/unstable, try reducing `w0` (e.g., 10–20) or rely on gradient clipping (already enabled in training)

### Scaling guidelines
- More complex potentials or tighter wells → consider increasing `hidden_features` or `hidden_layers`
- If residual stalls high and energy is off, scale `nq`/`nc` first (integration/collocation), then consider model capacity
- Balance model size with computational cost and training stability


## Usage in Training Code

The `src/train_1e.py` script instantiates SIREN by default:

```python
model = SIREN(
    in_features=2, 
    hidden_features=args.hidden, 
    hidden_layers=args.layers, 
    out_features=1
).to(device)
```

The architecture parameters are controlled by command-line arguments:
- `--hidden`: Sets `hidden_features` (default: 128)
- `--layers`: Sets `hidden_layers` (default: 6)

Note: The SIREN class constructor has `hidden_layers=4` as default, but the training script overrides this to 6.


## Notes on numerical stability

- Double precision: enabled globally to improve ∇²ψ
- Initialization: SIREN uses scale-aware init; avoid modifying unless you understand its effect on signal propagation
- Frequency scales: Extreme `w0` or `sigma` can lead to exploding/vanishing gradients
- Gradient clipping: training uses clipping to `max_norm=1.0` for robustness


## Summary

SIREN is specifically chosen for this quantum mechanics application because:
- Quantum eigenfunctions are inherently oscillatory, matching SIREN's sinusoidal nature
- The smooth periodic activations provide stable higher-order derivatives essential for accurate Laplacian computation
- Combined with double precision, SIREN delivers the numerical accuracy required for physics-informed neural networks solving eigenvalue problems

