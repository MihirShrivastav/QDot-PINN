# Loss Functions for Physics-Informed Neural Networks

This document explains all loss functions implemented in `src/pinn/losses.py` used to train neural networks for solving the 2D Schrödinger eigenvalue problem. The PINN approach combines multiple physics-motivated loss terms to learn wavefunctions ψ(x,y) and their corresponding energies E.

## Overview

The total loss function is a weighted combination of several terms:

```
L_total = λ_rr × E_RR + λ_pde × L_residual + λ_norm × L_norm + λ_ortho × L_ortho + λ_sym × L_symmetry
```

Each term serves a specific physical or mathematical purpose in constraining the neural network to learn valid quantum mechanical solutions.

## Core Loss Functions

### 1. Rayleigh-Ritz Energy (`rayleigh_ritz_energy`)

**Purpose**: Primary physics-based objective that minimizes the energy expectation value.

**Mathematical Form**:
```
E_RR(ψ) = [∫ (|∇ψ|² + v(x,y)|ψ|²) dΩ] / [∫ |ψ|² dΩ]
```

**Implementation Details**:
- Uses gradient-squared form `|∇ψ|²` for kinetic energy instead of integration by parts
- Ensures positive-definite kinetic term without boundary conditions
- Computed on structured quadrature grid (`xy_q`, `w_q`) for accurate integration
- Scale-invariant: ratio form makes it independent of wavefunction normalization

**Physical Significance**:
- Represents the energy expectation value ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩
- Minimizing this drives the network toward true eigenstates
- Variational principle guarantees ground state has lowest energy

**Parameters**:
- `fn`: Neural network function ψ(x,y)
- `vfun`: Potential function v(x,y)
- `xy_q`: Quadrature points [N_q, 2]
- `w_q`: Quadrature weights [N_q, 1]

**Usage**: Always included with weight `λ_rr` (typically 1.0 as reference scale)

### 2. PDE Residual Loss (`pde_residual_loss`)

**Purpose**: Enforces that the wavefunction satisfies the Schrödinger equation.

**Mathematical Form**:
```
L_residual = ⟨|Hψ - E_local × ψ|²⟩_collocation
```
where `E_local = ⟨ψ|Hψ⟩/⟨ψ|ψ⟩` is computed locally on collocation points.

**Implementation Details**:
- Applies Hamiltonian operator: `H = -∇² + v(x,y)`
- Projects out local energy estimate to avoid trivial ψ=0 solution
- Evaluated on random collocation points for computational efficiency
- Uses automatic differentiation for Laplacian computation

**Physical Significance**:
- Ensures the learned function actually solves the eigenvalue equation
- Complements energy minimization by enforcing PDE consistency
- Helps capture local wavefunction behavior between quadrature points

**Parameters**:
- `fn`: Neural network function ψ(x,y)
- `vfun`: Potential function v(x,y)
- `xy_c`: Collocation points [N_c, 2] (randomly sampled each epoch)

**Usage**: Included with weight `λ_pde` (typically 1.0, balance with energy term)

### 3. Orthogonality Loss (`orthogonality_loss`)

**Purpose**: Enforces orthogonality between excited states and lower-energy reference states.

**Mathematical Form**:
```
L_ortho = [⟨ψ_new, ψ_ref⟩ / (||ψ_new|| × ||ψ_ref||)]²
```

**Implementation Details**:
- Computes normalized overlap between current and reference wavefunctions
- Uses quadrature grid for accurate inner product computation
- Reference wavefunction is frozen (no gradients)
- Squared to ensure positive penalty

**Physical Significance**:
- Quantum mechanical eigenstates must be orthogonal
- Prevents excited state collapse to ground state
- Implements deflation method for finding higher eigenstates

**Parameters**:
- `psi_new`: Current wavefunction values [N, 1]
- `psi_prev`: Reference wavefunction values [N, 1] (frozen)
- `w_q`: Quadrature weights [N, 1]

**Usage**: Only for excited states (`--state > 0`) with weight `λ_ortho` (typically 10.0)

## Auxiliary Loss Functions

### 4. Normalization Penalty

**Purpose**: Maintains unit normalization of the wavefunction.

**Mathematical Form** (implemented in `train_1e.py`):
```
L_norm = (∫ |ψ|² dΩ - 1)²
```

**Implementation Details**:
- Computed on quadrature grid for accurate integration
- Prevents trivial ψ=0 solution
- Helps stabilize training dynamics

**Physical Significance**:
- Quantum mechanical wavefunctions must be normalized
- Ensures probability interpretation: ∫|ψ|² = 1
- Provides scale reference for the wavefunction

**Usage**: Included with weight `λ_norm` (typically 1.0, increase if normalization drifts)

### 5. Symmetry Penalties

#### Even Parity (`symmetry_penalty_even`)

**Purpose**: Enforces even symmetry ψ(x,y) = ψ(-x,y) for symmetric ground states.

**Mathematical Form**:
```
L_sym_even = ⟨(ψ(x,y) - ψ(-x,y))²⟩_w
```

**Physical Significance**:
- Ground state of symmetric double dot (δ=0) has even parity
- Bonding orbital with no node at x=0
- Helps convergence to correct symmetry class

#### Odd Parity (`symmetry_penalty_odd`)

**Purpose**: Enforces odd symmetry ψ(x,y) = -ψ(-x,y) for antisymmetric excited states.

**Mathematical Form**:
```
L_sym_odd = ⟨(ψ(x,y) + ψ(-x,y))²⟩_w
```

**Physical Significance**:
- First excited state of symmetric double dot (δ=0) has odd parity
- Antibonding orbital with node at x=0
- Distinguishes excited state from ground state

**Parameters** (both symmetry functions):
- `fn`: Neural network function ψ(x,y)
- `xy_q`: Quadrature points [N, 2]
- `w_q`: Quadrature weights [N, 1]

**Usage**: Optional, only for symmetric cases (δ≈0) with small weights (0.05-0.3)

## Utility Functions

### `laplacian`

**Purpose**: Computes the Laplacian ∇²ψ using automatic differentiation.

**Implementation**:
- Takes second derivatives with respect to x and y coordinates
- Uses `torch.autograd.grad` with `create_graph=True` for higher-order derivatives
- Essential for kinetic energy and Hamiltonian operator

### `h_psi`

**Purpose**: Applies the Hamiltonian operator H = -∇² + v(x,y) to the wavefunction.

**Implementation**:
- Combines Laplacian (kinetic) and potential (multiplicative) terms
- Used in PDE residual computation
- Represents the left-hand side of the Schrödinger equation

## Loss Balancing Strategy

### Typical Weight Values:
- `λ_rr = 1.0`: Reference scale for energy minimization
- `λ_pde = 1.0`: Equal importance to energy term
- `λ_norm = 1.0-100.0`: Increase if normalization drifts
- `λ_ortho = 10.0`: Strong orthogonality enforcement for excited states
- `λ_sym = 0.05-0.3`: Weak symmetry guidance (only when needed)

### Balancing Considerations:
1. **Energy vs. PDE**: Balance physics (energy) with mathematical constraint (PDE)
2. **Normalization**: Increase weight if ∫|ψ|² drifts far from 1
3. **Orthogonality**: Strong enough to prevent collapse, not so strong as to dominate
4. **Symmetry**: Light touch to guide, not force, correct parity

## Training Dynamics

### Early Training:
- Energy and normalization terms dominate
- PDE residual typically high initially
- Gradual satisfaction of all constraints

### Convergence:
- Energy reaches variational minimum
- PDE residual decreases (target: 1e-2 to 1e-1)
- Normalization stabilizes near 1.0
- Orthogonality (if applicable) approaches zero

### Monitoring:
- Track all loss components separately in training history
- Watch for loss component imbalances
- Adjust weights if one term dominates or becomes negligible

## Computational Considerations

### Automatic Differentiation:
- All functions use `torch.autograd` for derivatives
- `create_graph=True` enables higher-order derivatives
- `retain_graph=True` allows multiple backward passes

### Numerical Stability:
- Small epsilon values (1e-12) prevent division by zero
- Double precision (`torch.float64`) improves derivative accuracy
- Gradient clipping prevents exploding gradients

### Memory Usage:
- Quadrature grids stored in memory for repeated use
- Collocation points regenerated each epoch
- Reference wavefunctions cached for orthogonality

## Usage Examples

### Ground State Training:
```python
# Primary losses only
loss = λ_rr * rayleigh_ritz_energy(model, vfun, xy_q, w_q) + \
       λ_pde * pde_residual_loss(model, vfun, xy_c) + \
       λ_norm * normalization_penalty(model, xy_q, w_q)
```

### Excited State Training:
```python
# Add orthogonality constraint
loss = λ_rr * rayleigh_ritz_energy(model, vfun, xy_q, w_q) + \
       λ_pde * pde_residual_loss(model, vfun, xy_c) + \
       λ_norm * normalization_penalty(model, xy_q, w_q) + \
       λ_ortho * orthogonality_loss(psi_new, psi_ref, w_q)
```

### Symmetric Case with Parity:
```python
# Add symmetry guidance
loss += λ_sym_even * symmetry_penalty_even(model, xy_q, w_q)  # Ground state
# OR
loss += λ_sym_odd * symmetry_penalty_odd(model, xy_q, w_q)    # Excited state
```