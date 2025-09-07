from __future__ import annotations

"""
Finite-difference baseline for 1e in 2D: solve (-∇^2 + v) psi = E psi on a box with Dirichlet BC.
Requires numpy and (optionally) scipy.sparse for efficiency.
"""

import numpy as np
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:  # pragma: no cover
    sp = None
    spla = None


def build_hamiltonian(nx: int, ny: int, X: float, Y: float, v_grid: np.ndarray):
    dx = 2 * X / (nx - 1)
    dy = 2 * Y / (ny - 1)
    N = nx * ny
    if sp is None:
        raise RuntimeError("scipy.sparse not available; install scipy to run FDM baseline.")

    main = np.full(N, 2.0 / (dx * dx) + 2.0 / (dy * dy)) + v_grid.reshape(-1)
    offx = np.full(N - 1, -1.0 / (dx * dx))
    offy = np.full(N - nx, -1.0 / (dy * dy))

    # 2D 5-point Laplacian with Dirichlet at boundary (implicitly handled by not connecting across rows)
    diags = [main, offx, offx, offy, offy]
    offsets = [0, -1, 1, -nx, nx]
    A = sp.diags(diags, offsets, shape=(N, N), format="csr")

    # Zero out connections across row boundaries
    for i in range(1, ny):
        idx = i * nx
        A[idx, idx - 1] = 0.0
        A[idx - 1, idx] = 0.0

    return A


def solve_lowest(nx: int, ny: int, X: float, Y: float, v_func):
    xs = np.linspace(-X, X, nx)
    ys = np.linspace(-Y, Y, ny)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    v_grid = np.vectorize(v_func)(Xg, Yg)

    H = build_hamiltonian(nx, ny, X, Y, v_grid)
    k = 4  # lowest few states
    vals, vecs = spla.eigs(H, k=k, which="SR")  # smallest real parts
    order = np.argsort(vals.real)
    return vals.real[order], vecs[:, order].real, (xs, ys)

