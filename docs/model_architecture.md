# Model Architecture: Networks used to approximate ψ(x, y)

This document explains the neural architectures implemented in `src/pinn/models.py` that approximate the real-valued wavefunction ψ(x, y) for the 2D, single-electron Schrödinger eigenproblem.

We currently provide three variants:
- SIREN: Sinusoidal Representation Network (default)
- MLP: Standard multilayer perceptron with configurable activation (default: Tanh)
- MLPFourier: Random Fourier feature encoder + MLP head

All models take 2D inputs (x, y) and output a single scalar ψ. They are differentiable end-to-end; we compute ∇ψ, ∇²ψ via autograd for losses.


## 1) SIREN (Sinusoidal Representation Network)

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


## 2) MLP (Tanh by default)

A standard MLP with configurable activation. It is simpler but has spectral bias (tends to learn low frequencies first).

- Module: `MLP`
- Constructor signature:
  - `MLP(in_features: int, hidden_features: int = 128, hidden_layers: int = 4, out_features: int = 1, activation: nn.Module | None = None)`
  - Default activation is Tanh; you can pass `nn.SiLU()`, `nn.ReLU()`, etc.
- Architecture: `[Linear → Act] × hidden_layers` with matching widths, then a final Linear to 1D

When to consider MLP
- For very smooth, slowly-varying potentials or coarse approximations
- When you prefer simpler numerics (no sine frequencies to tune)

Caveats
- May struggle to capture high-frequency detail (nodes, interference fringes) unless made very deep/wide or combined with Fourier features


## 3) MLPFourier (Random Fourier features + MLP)

Encodes inputs with fixed random Fourier features z = [sin(2πxB), cos(2πxB)] before passing to an MLP.

- Modules: `FourierFeatures`, `MLPFourier`
- FourierFeatures
  - `FourierFeatures.random(in_features: int, n_features: int = 64, sigma: float = 10.0)`
  - Stores a matrix `B ∈ R^{in_features×n_features}`; not a learnable parameter by default (dataclass field)
  - Encoding: `z = [sin(2π x B), cos(2π x B)] ∈ R^{2·n_features}`
- MLPFourier
  - `MLPFourier(in_features: int, ff: FourierFeatures, hidden_features: int = 128, hidden_layers: int = 4, out_features: int = 1)`
  - Internally builds `MLP(in_features=2·n_features, ...)`

When to use
- If you want an alternative to SIREN but still need higher-frequency capacity
- `sigma` controls the frequency band. Larger `sigma` ⇒ higher frequency embeddings (can be harder to optimize)


## Shapes, dtypes, and derivatives

- Inputs: `[N, 2]` float tensor with coordinates (x, y)
- Outputs: `[N, 1]` scalar ψ values
- Dtype: we set default to float64 (double). This improves stability for second derivatives
- Derivatives: we rely on autograd for ∇ψ and ∇²ψ (see `src/pinn/losses.py`). When computing derivatives, make sure `xy.requires_grad_(True)` is set as done in the loss functions


## Choosing an architecture

Recommended defaults
- Use SIREN for most runs: `hidden=128, layers=4, w0=30.0, w0_hidden=1.0`
- If training is noisy/unstable, try reducing `w0` (e.g., 10–20) or gradient clipping (already enabled in training)
- If you prefer ReLU/Tanh MLPs, start with the same width/depth and consider adding Fourier features if details are missing

Heuristics
- More complex potentials or tighter wells ⇒ consider increasing width/layers
- If residual stalls high and energy is off, scale `nq`/`nc` first (integration/collocation), then consider model capacity


## How training code picks the model

Currently, `src/train_1e.py` instantiates `SIREN` by default:


````python
model = SIREN(in_features=2, hidden_features=args.hidden, hidden_layers=args.layers, out_features=1).to(device)
````

To try `MLP` instead (example):


````python
from src.pinn.models import MLP
model = MLP(in_features=2, hidden_features=128, hidden_layers=4, out_features=1, activation=nn.Tanh()).to(device)
````


To try `MLPFourier` (example):


````python
from src.pinn.models import FourierFeatures, MLPFourier
ff = FourierFeatures.random(in_features=2, n_features=64, sigma=10.0, device=device)
model = MLPFourier(in_features=2, ff=ff, hidden_features=128, hidden_layers=4, out_features=1).to(device)
````


## Notes on numerical stability

- Double precision: enabled globally to improve ∇²ψ
- Initialization: SIREN uses scale-aware init; avoid modifying unless you understand its effect on signal propagation
- Frequency scales: Extreme `w0` or `sigma` can lead to exploding/vanishing gradients
- Gradient clipping: training uses clipping to `max_norm=1.0` for robustness


## Summary

- SIREN is the default and best starting point for oscillatory quantum eigenstates
- MLP is simpler but more biased to low frequencies; combine with Fourier features for more expressiveness
- All models are differentiable, producing stable first/second derivatives required by the loss terms (Rayleigh quotient and PDE residual)

