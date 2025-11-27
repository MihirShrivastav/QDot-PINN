# PINN Solver for Double Quantum Dot Eigenstates

This repository implements a physics-informed neural solver for the 2D, single-electron Schrödinger eigenproblem in GaAs double quantum dots (DQDs). The code base is split between a modular Python package (`src/`) and a self-contained Jupyter/Colab notebook (`train_dqd_colab.ipynb`). Both variants use sinusoidal representation networks (SIREN) trained with a Rayleigh–Ritz objective, PDE residual penalties, and orthogonality constraints to resolve the ground and first two excited states without meshing.

---

## Physical Problem

### Device motivation
- **Spin qubits**: Gate-defined GaAs DQDs confine a single electron in two adjacent minima; the energy splitting between bonding/antibonding states sets the tunnel coupling \(2t\) that enters the singlet–triplet qubit Hamiltonian.
- **Soft confinement**: Fabricated devices never produce infinite square wells; electrostatic gates yield approximately quartic wells with finite walls. Capturing the correct curvature near each minimum is essential for matching measured spectra.
- **Dimensional reduction**: We work in a 2D effective-mass formulation where lateral confinement dominates (vertical confinement is treated as frozen). The problem therefore reduces to solving the stationary Schrödinger equation in a rectangular domain.

### Governing equation
\[
\left[-\frac{\hbar^2}{2 m^\*}(\partial_{xx} + \partial_{yy}) + V(x,y)\right] \psi(x,y) = E \psi(x,y), \qquad (x,y) \in [-X,X]\times[-Y,Y].
\]
The envelope function \(\psi\) is normalised (\(\int |\psi|^2 = 1\)) and must decay near the domain boundary. We embed the material parameters through dimensionless scaling so that the learned eigenvalues can be mapped back to meV via \(E_0 = 0.6318\) meV for GaAs (\(m^\*=0.067 m_0\)).

### Potential parameters and their physical role
- **\(a\)**: controls the separation between the left and right wells; increasing \(a\) lowers tunnel coupling and modifies the splitting between GS and ES1.
- **\(c_4\)** / **\(\hbar\omega_x\)**: set the quartic curvature along \(x\). We usually specify the desired harmonic energy \(\hbar\omega_x\) and convert it to \(c_4\) so that the Taylor expansion around each minimum matches the measured orbital spacing.
- **\(c_{2y}\)** / **\(\hbar\omega_y\)**: determine confinement along \(y\). Larger values mimic a tighter electrostatic channel and raise the energy of vertically excited states (e.g., ES2 in our reference run).
- **\(\Delta\)**: a linear detuning term that models differential gate voltages between the two dots; turning it on breaks parity and shifts the charge distribution.

### Why these loss terms?
- **Rayleigh–Ritz energy** \(E_{\text{RR}}\): Provides a variational upper bound on the true eigenvalue and enforces global behaviour. Minimising this term ensures accurate energies even if the PDE residual momentarily fluctuates.
- **PDE residual** \(L_{\text{res}}\): Enforces the Schrödinger equation locally at collocation points, guaranteeing that \(\psi\) satisfies the differential operator, especially in regions where the Rayleigh quotient alone could be satisfied with the wrong shape.
- **Normalization penalty** \(L_{\text{norm}}\): Maintains \(\int |\psi|^2 = 1\), which fixes the scale of \(\psi\) and stabilises the Rayleigh quotient.
- **Orthogonality penalty** \(L_{\text{ortho}}\): Enforces \(\langle \psi_n | \psi_m \rangle = 0\) for \(n\ne m\), allowing consecutive excited states to be trained sequentially without collapsing onto previously discovered modes.
- **Parity penalties** \(L_{\text{sym-even}}, L_{\text{sym-odd}}\): For symmetric potentials (\(\Delta=0\)), small penalties accelerate convergence to the expected even/odd modes, reducing the chance of hybridised solutions when eigenvalues are clustered.

These terms are weighted via `--lam-*` hyperparameters so that each contribution reflects its physical priority (e.g., Rayleigh–Ritz dominates energy accuracy, while a moderate `lam_pde` enforces local fidelity). The values in `results/` were tuned to balance residuals in the \(10^{-2}\)–\(10^{-3}\) range with stable normalization.

### Outputs
- Energy spectrum in both dimensionless units and meV.
- Wavefunction grids, density plots, and derived observables such as center of mass, side masses, and tunnel splitting.
`results/gs`, `results/es1`, and `results/es2` illustrate the expected directory structure for a complete eigenstate characterization.

---

## PINN Formulation

- **Architecture**: 6 hidden layers × 128 neurons, sine activations, double precision for stable Laplacians (see `src/pinn/models.py`).
- **Loss terms**:
  - Rayleigh–Ritz energy (global accuracy/bounds).
  - PDE residual on randomly sampled collocation points.
  - Normalization penalty.
  - Orthogonality + optional parity penalties for excited states.
- **Optimization**: Adam (2k epochs in notebook setting) + L-BFGS refinement.
- **Diagnostics**: history logging, density features, potential snapshots, and saved grids of \(\psi(x,y)\).

---

## Repository Layout

- `src/` – Training scripts, physics helpers, SIREN modules, visualization utilities.
- `train_dqd_colab.ipynb` – Self-contained workflow (imports, training, diagnostics, artifact saving).
- `results/` – Reference experiment exported from Colab (see below).
- `docs/` – Additional problem statements (`docs/physics_problem.md`, etc.).
- `data/` – Default CLI outputs.

---

## Running the Code

### Python package / CLI

1. Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install torch numpy scipy matplotlib seaborn tqdm
   ```
2. (Optional) enable editable installs for easier imports:
   ```bash
   pip install -e .
   ```
3. Launch training from the repo root:
   ```bash
   python -m src.train_1e \
     --state 0 \
     --outdir data/run_1e_gs \
     --a 1.5 --hbar-omega-x 3.0 --hbar-omega-y 5.0 \
     --nq 128 --nc 8192 --epochs 1200 --lbfgs-iters 200
   ```
   - Add `--state 1 --ref-model <gs path> --lam-ortho 10 --lam-sym-odd 0.1` for the first excited state.
   - Use `--state 2 --ref-models <gs> <es1> --lam-sym-even 0.1` for the second excited state.
   - `--viz-only` regenerates diagnostics without retraining.

### Notebook / Colab workflow

- Open `train_dqd_colab.ipynb` locally (JupyterLab) or in Google Colab (GPU optional).
- Run the notebook top-to-bottom. Each state (GS, ES1, ES2) writes a dedicated folder in `colab_results/` with:
  - `model_best.pt`, `config.json`, `params.txt`
  - `energies.txt`, `history.{json,csv}`, `density_features.{json,csv}`
  - Saved plots: density, training curves, potential snapshots, combined figure
  - `psi_grid.pt` (torch) and NumPy exports for downstream analysis
- To rerun on Colab, upload the notebook and execute **Runtime → Run all**. The exported artifacts used in this README live under `results/`.

---

## Reference Experiment (`results/`)

We ran the notebook on Colab with GaAs parameters \(a=1.5\), \(\hbar\omega_x=3\) meV, \(\hbar\omega_y=5\) meV, \(\Delta=0\), domain \([-4,4]^2\), \(n_q=128\), \(n_c=8192\), 2000 Adam epochs, and 200 L-BFGS iterations. The produced artifacts are versioned in `results/`, mirroring the structure we expect from anyone running the notebook.

### Potential inspection

- `results/potential/potential_density.png` – contour view of the biquadratic landscape.
- `results/potential/potential_3d_surface.png` – height-map for the double-well geometry.

### Eigenvalue summary

| State | \(E\) (dimensionless) | \(E\) (meV) | Splitting to previous state (meV) |
|-------|----------------------:|------------:|----------------------------------:|
| GS    | 5.9129               | 3.736       | — |
| ES1   | 6.2915               | 3.975       | 0.239 (ΔE\_{ES1−GS}) |
| ES2   | 8.9098               | 5.630       | 1.654 (ΔE\_{ES2−ES1}) |

Source files: `results/gs/energy_gs.json`, `results/es1/energy_es1.json`, `results/es2/energy_es2.json`.

### Spatial diagnostics

- **Ground state** (`results/gs/gs_density.png`): symmetric bonding mode with center-of-mass at (0.0029, −0.0399) and nearly equal left/right probability (0.498 vs 0.502).
- **First excited** (`results/es1/es1_density.png`): odd-parity state with COM ≈ (0.0008, 0.022). Density peaks straddle x = 0 with equal mass sharing.
- **Second excited** (`results/es2/es2_density.png`): even parity along x with a node at y ≈ 0, COM ≈ (0.0010, 0.0036), balanced side mass.

Each folder contains `density_features.json` (probability, COM, side masses, top peaks, minima sampling) and `training_history.json` for reproducibility. The saved grids (`*_wavefunction_psi_grid.npy` + `xs/ys`) enable downstream Fourier analysis or overlap computations.

### Training traces

- Histories saved as PNG + JSON in each state directory show Rayleigh–Ritz energy decay, PDE residual convergence (~10⁻² to 10⁻³), normalization stability, and orthogonality metrics (Overlap0 plots).
- Combined comparison plot: `results/all_states.png`.

---

## How to Reproduce the Reference Results

1. Clone the repo and open `train_dqd_colab.ipynb` in Colab.
2. Update the output root in the configuration cell if you want a different folder name.
3. Run all cells. Three state directories mirror those in `results/`, so you can diff outputs against the provided artifacts.
4. Optionally download the saved `.pt` weights and plug them into `src.train_1e --viz-only` for validation on your workstation.

---

## Interpreting / Extending the Data

- Use `results/*/config.json` and `params.txt` to capture every hyperparameter and potential coefficient of the run.
- `history.json` files allow plotting convergence statistics across experiments.
- The `potential/` figures provide context when changing \(a\), \(\hbar\omega_x\), \(\hbar\omega_y\), or detuning.
- To explore different excited states or asymmetries, modify the notebook configuration cell, rerun, and compare against the baseline figures above.

---

## Citation

```
@misc{pinn_dqd_2025,
  title = {Physics-Informed Neural Solver for Double Quantum Dot Eigenstates},
  author = {Mihir Shrivastava},
  year = {2025},
  howpublished = {https://github.com/MihirShrivastav/QDot-PINN},
  note = {SIREN PINN implementation for GaAs DQDs}
}
```

Questions or collaboration ideas? Please open an issue in this repository.
