# Self-Contained Colab Notebook for DQD PINN Training

## Overview

`train_dqd_colab.ipynb` is a **completely self-contained** Jupyter notebook for training Physics-Informed Neural Networks (PINNs) to solve the single-electron Schrödinger eigenproblem in a 2D double quantum dot (GaAs).

**Key feature**: All code is embedded directly in the notebook—no external dependencies on project files.

## What's Included

The notebook contains **16 cells** organized sequentially:

### Setup (Cells 1-7)
1. **Imports and Constants** - PyTorch, physical constants (ℏ, e, m₀, etc.)
2. **Physics Classes** - GaAs material parameters, BiquadraticParams, potential functions
3. **SIREN Neural Network** - Sine activation network architecture
4. **Loss Functions** - Rayleigh-Ritz energy, PDE residual, orthogonality, parity penalties
5. **Helper Functions** - Quadrature grids, collocation sampling, density analysis
6. **Training Loop** - Complete Adam + LBFGS training with all constraints
7. **Visualization Functions** - Density plots, training curves

### Configuration (Cell 8)
8. **Configuration** - All hyperparameters in one place:
   - Material: GaAs (L₀=30 nm, E₀≈0.632 meV)
   - Potential: a=1.5, ℏωₓ=3 meV, ℏωᵧ=5 meV, δ=0
   - Training: 1000 Adam epochs + 200 LBFGS iterations
   - Loss weights: λ_norm=100, λ_pde=2.0, λ_ortho=20
   - Parity: λ_sym_even=0.1 (GS, ES2), λ_sym_odd=0.1 (ES1)

### Training & Visualization (Cells 9-16)
9. **Train Ground State (GS)** - With even-parity bias
10. **Visualize GS** - Density plot, training curves, features
11. **Train First Excited State (ES1)** - Orthogonal to GS, odd-parity bias
12. **Visualize ES1** - Density plot, training curves, features
13. **Train Second Excited State (ES2)** - Orthogonal to GS & ES1, even-parity bias
14. **Visualize ES2** - Density plot, training curves, features
15. **Summary** - Energy table, side-by-side comparison plot

## How to Use on Google Colab

### Option 1: Upload Directly
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `train_dqd_colab.ipynb`
4. Run cells sequentially (Runtime → Run all)

### Option 2: From GitHub
1. Upload `train_dqd_colab.ipynb` to your GitHub repo
2. In Colab: **File → Open notebook → GitHub**
3. Enter your repo URL
4. Run cells sequentially

## Expected Runtime

- **CPU**: ~15-20 minutes total (GS: 5 min, ES1: 6 min, ES2: 7 min)
- **GPU (T4)**: ~5-8 minutes total (GS: 2 min, ES1: 2.5 min, ES2: 3 min)

The notebook automatically detects and uses GPU if available.

## Output

All results are saved to `colab_results/` directory:
- `model_gs.pt`, `model_es1.pt`, `model_es2.pt` - Trained model weights
- `energies.txt` - Energy eigenvalues (dimensionless + meV)
- `all_states.png` - Side-by-side comparison of GS, ES1, ES2

## Customization

To modify training parameters, edit **Cell 8 (Configuration)**:

```python
# Example: Train with stronger orthogonality constraint
lam_ortho = 30.0  # default: 20.0

# Example: Longer training
epochs = 2000  # default: 1000
lbfgs_iters = 300  # default: 200

# Example: Different potential (asymmetric)
delta = 0.2  # default: 0.0 (symmetric)
```

## Physics Background

- **Problem**: Solve -∇²ψ + v(x,y)ψ = Eψ for biquadratic potential v(x,y) = c₄(x²-a²)² + c₂ᵧy² + δx
- **Method**: Minimize Rayleigh quotient E[ψ] = ∫(|∇ψ|² + vψ²)/∫|ψ|² with PDE residual regularization
- **Constraints**: Normalization (∫|ψ|²=1), orthogonality (⟨ψᵢ|ψⱼ⟩=0), parity (even/odd symmetry)
- **Network**: SIREN (sine activations) with 4 hidden layers, 128 units each
- **Optimizer**: Adam (1000 epochs) → LBFGS (200 iterations)

## Expected Results (δ=0, symmetric case)

- **GS**: Even parity, two lobes at (±a, 0), E_GS ≈ 1.5-2.0 (dimensionless)
- **ES1**: Odd parity, node at x=0, E_ES1 > E_GS, splitting ΔE ≈ 0.3-0.5
- **ES2**: Even parity, higher energy, E_ES2 > E_ES1

## Troubleshooting

**Issue**: "CUDA out of memory"
- **Fix**: Reduce `nq` (128 → 96) or `nc` (8192 → 4096) in Cell 8

**Issue**: Poor orthogonality (Overlap0 > 0.01)
- **Fix**: Increase `lam_ortho` (20 → 30) or `epochs` (1000 → 1500)

**Issue**: Wavefunctions not normalized (|ψ|² ≠ 1)
- **Fix**: Increase `lam_norm` (100 → 200)

**Issue**: Wrong parity (e.g., ES1 looks even)
- **Fix**: Increase `lam_sym_odd` (0.1 → 0.3) for ES1

## Notes

- The notebook uses **double precision** (float64) for accurate second derivatives
- All code is self-contained—no imports from `src/` modules
- Results are reproducible (PyTorch default seed)
- Gradient clipping (max_norm=1.0) prevents instabilities

## Citation

If you use this notebook, please cite:
```
@misc{dqd_pinn_2024,
  title={Physics-Informed Neural Networks for Double Quantum Dot Eigenstates},
  author={Your Name},
  year={2024},
  note={Self-contained Colab notebook for GaAs DQD PINN training}
}
```

---

**Questions?** Check the main project README or open an issue on GitHub.

