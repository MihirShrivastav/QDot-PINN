# PINN Solver for Double Quantum Dot Eigenstates

## Abstract
This repository implements a physics-informed neural solver for the two-dimensional, single-electron Schrodinger eigenvalue problem in GaAs double quantum dots. The training loop works in dimensionless units, embeds the GaAs material parameters, and targets the low-lying bound states that determine tunnelling rates and charge localization in spin-qubit devices. The solver is purpose-built for oscillatory quantum states: it relies on sinusoidal representation networks (SIREN), a Rayleigh-Ritz variational loss, and local PDE residual constraints. The same formulation is available both as modular Python modules (`src/`) and as a standalone Colab notebook (`train_dqd_colab.ipynb`).

## Scientific Motivation

- **Device design**: Accurate estimates of tunnel coupling, detuning response, and spatial localization are needed when engineering GaAs double quantum dots for singlet-triplet qubits.
- **Soft potentials**: Experimental devices rarely realize hard-wall potentials; using smooth biquadratic wells makes soft boundaries an explicit modelling choice rather than a nuisance.
- **Mesh-free eigenproblem**: Classical solvers require costly remeshing whenever the potential changes. The PINN formulation handles new potentials or detuning values by retraining a continuous network.

## Governing Problem
We solve the stationary Schrodinger equation in a finite rectangle `[-X, X] x [-Y, Y]`. The implemented biquadratic potential is
```
V(x,y) = c4 (x^2 - a^2)^2 + c2y y^2 + delta * x,
```
where `a` controls dot separation, `c4` and `c2y` set the curvatures (or equivalently the target harmonic energies `--hbar-omega-x` and `--hbar-omega-y`), and `delta` applies an electric-field detuning. `src/physics/units.py` provides the GaAs scaling (length 30 nm, energy 0.6318 meV), so energies reported by the solver can be converted back to physical units by multiplication.

## Method Overview

### SIREN PINN architecture
- Inputs: coordinates `(x, y)` in dimensionless form.
- Model: depth 6, width 128 sine-activated layers (`src/pinn/models.py`).
- Output: scalar wavefunction `psi(x, y)` evaluated in double precision to stabilise second derivatives.

### Loss formulation
- **Rayleigh-Ritz quotient** for global eigenvalue accuracy.
- **PDE residual** on random collocation points to ensure local equation satisfaction.
- **Normalization** term to keep the integrated probability equal to one.
- **Orthogonality** penalties against previously trained states when solving for excited modes.
- **Parity priors** (optional) when the device is symmetric (`delta = 0`).

### Training strategy
1. **Adam phase** (default 1000 epochs) with clipped gradients for coarse convergence.
2. **L-BFGS refinement** (default 200 steps) for sharp minima.
3. **Diagnostics**: energy history, residual decay, normalization drift, and overlap with reference states are logged every epoch.

### Outputs and diagnostics
Each training run (see `--outdir`) stores
- `model_best.pt`, `config.json`, and `params.txt` for reproducibility.
- `energies.txt`, `history.{json,csv}`, and `density_features.{json,csv}`.
- Visualization assets: `psi_density.png`, `potential.png`, and optional training curves.
- `psi_grid.pt` for downstream numerical analysis.

## Repository Layout
- `src/` - Core PINN implementation (training entry points, models, physics helpers, visualization utilities).
- `train_dqd_colab.ipynb` - Self-contained notebook that re-implements the entire solver in 16 cells.
- `COLAB_NOTEBOOK_README.md` - Additional background on the notebook flow.
- `data/` - Default location for training outputs.
- `docs/` - Extended documentation and figures.

## Environment and Setup
1. Install Python 3.10+ and PyTorch 2.x (CUDA optional).
2. Create an isolated environment, e.g.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install torch numpy scipy matplotlib seaborn tqdm
   ```
3. (Optional) Install the repo as a package to enable module imports:
   ```bash
   pip install -e .
   ```
   If you skip this step, run CLI commands with `python -m src.train_1e ...` from the repository root so relative imports resolve.

## Command-Line Workflow
All commands are issued from the repository root.

### Ground state (state = 0)
```bash
python -m src.train_1e   --outdir data/run_1e_gs   --device cuda   --nq 128   --nc 4096   --epochs 1200   --lbfgs-iters 200
```
Key options:
- Potential: either supply `--a --hbar-omega-x --hbar-omega-y --delta` or give explicit `--c4 --c2y --delta`.
- Loss weights: `--lam-rr`, `--lam-pde`, `--lam-norm` tune the balance between Rayleigh energy and residual terms.
- Numerical quadrature: `--nq` sets the Gauss-Lobatto grid per axis; `--nc` controls random collocation samples.

### First excited state (state = 1)
```bash
python -m src.train_1e   --state 1   --ref-model data/run_1e_gs/model_best.pt   --lam-ortho 10.0   --lam-sym-odd 0.1   --outdir data/run_1e_es1
```
Provide the ground-state weights via `--ref-model` (or multiple references using `--ref-models`) so orthogonality and parity terms suppress mode collapse. Higher excited states reuse the same command with additional references.

### Visualization-only regeneration
```bash
python -m src.train_1e --viz-only --outdir data/run_1e_gs
```
This reloads `model_best.pt`, recomputes energy diagnostics, and writes `energies_eval.txt`, updated density maps, and grid data without retraining. You may override the weights path with `--model path/to/model.pt`.

## Consolidated Notebook Workflow
`train_dqd_colab.ipynb` contains the entire pipeline-imports, model, training loops, diagnostics, and visualization-embedded inside a single notebook. Use it when you want a one-file experiment (for example on Google Colab) without cloning the repo.

### Running on Google Colab
1. Visit <https://colab.research.google.com/>.
2. Upload `train_dqd_colab.ipynb` (File -> Upload notebook) or open it directly from GitHub (File -> Open notebook -> GitHub).
3. Enable a GPU runtime (Runtime -> Change runtime type -> GPU) for faster execution.
4. Execute **Runtime -> Run all**. The notebook trains ground, first excited, and second excited states sequentially, saving artifacts in `colab_results/` alongside comparison plots.

### Running locally (JupyterLab)
1. Install JupyterLab inside your environment (`pip install jupyterlab`).
2. Launch `jupyter lab`, open `train_dqd_colab.ipynb`, and run the cells in order. All dependencies (PyTorch, numpy, matplotlib) are installed directly within the notebook, so no project modules are required.

### Notebook contents
- Cells 1-5: physics constants, SIREN definition, and helper utilities.
- Cells 6-8: configuration cell exposing all hyperparameters (epochs, quadrature size, penalty weights, detuning, etc.).
- Cells 9-16: sequential training/visualization for the ground state, first excited, and second excited states, plus an aggregated summary table and combined plot.

## Interpreting Results
- **Energy metrics**: `energies.txt` and `energies_eval.txt` list both instantaneous and best-observed Rayleigh quotients. Values are dimensionless but can be multiplied by 0.6318 meV to map back to GaAs units.
- **Density diagnostics**: `density_features.*` captures center of mass, left/right dot probability, and dominant peaks to quantify localization and tunnel coupling.
- **Training history**: inspect `history.csv` to verify residual convergence, normalization stability, and orthogonality (Overlap0 column) for excited states.

## Reproducibility
- All random sampling uses PyTorch's RNG; set `PYTORCH_SEED` before launching a run to make collocation draws deterministic.
- `config.json` stores the exact CLI arguments, potential parameters, and quadrature settings per run.
- The notebook version prints the configuration dictionary before each phase, making it easy to cross-reference with command-line experiments.

## Citation
If this solver contributes to your research, please cite it as
```
@misc{pinn_dqd_2024,
  title={Physics-Informed Neural Solver for Double Quantum Dot Eigenstates},
  author={Mihir Shrivastava},
  year={2025},
  howpublished={https://github.com/MihirShrivastav/QDot-PINN},
  note={SIREN PINN implementation for GaAs DQDs}
}
```

For technical questions or collaborations, please open an issue or reach out via the project repository.
