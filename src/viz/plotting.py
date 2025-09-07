from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_potential(v_func, X: float, Y: float, nx: int = 200, ny: int = 200, title: str = "Potential v(x,y)"):
    xs = np.linspace(-X, X, nx)
    ys = np.linspace(-Y, Y, ny)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    V = np.vectorize(v_func)(Xg, Yg)
    plt.figure(figsize=(5, 4))
    im = plt.pcolormesh(Xg, Yg, V, shading="auto", cmap="viridis")
    plt.colorbar(im, label="v(x,y)")
    plt.title(title)
    plt.xlabel("x (dimless)")
    plt.ylabel("y (dimless)")
    plt.tight_layout()
    return plt.gcf()


def plot_wavefunction_density(
    xs,
    ys,
    psi_vec,
    nx: int,
    ny: int,
    title: str = r"$|\psi|^2$",
    *,
    cmap: str = "inferno",
    gamma: float = 0.6,
    overlay_contours: bool = True,
    contour_levels: int | None = 12,
    overlay_points: list[tuple[float, float]] | None = None,
    aspect: str = "equal",
):
    """
    Plot |psi|^2 with improved visibility and optional overlays.
    - gamma: <1 brightens low intensities for faint lobes
    - overlay_points: list of (x,y) to mark (e.g., potential minima at (±a,0))
    """
    P = (psi_vec.reshape(ny, nx) ** 2)
    plt.figure(figsize=(6, 5))
    norm = mcolors.PowerNorm(gamma=gamma) if gamma is not None else None
    im = plt.pcolormesh(xs, ys, P, shading="auto", cmap=cmap, norm=norm)
    plt.colorbar(im, label=r"$|\psi|^2$")
    if overlay_contours:
        try:
            cs = plt.contour(xs, ys, P, levels=contour_levels or 10, colors="w", linewidths=0.6, alpha=0.8)
            plt.clabel(cs, inline=True, fontsize=7, fmt="%1.2f")
        except Exception:
            pass
    if overlay_points:
        for (px, py) in overlay_points:
            plt.plot([px], [py], marker="o", markersize=4, color="cyan", mec="k", mew=0.3, alpha=0.9)
    plt.gca().set_aspect(aspect)
    plt.title(title)
    plt.xlabel("x (dimless)")
    plt.ylabel("y (dimless)")
    plt.tight_layout()
    return plt.gcf()



def plot_training_curves(history: dict, out_path: str | None = None):
    import numpy as np
    E = np.array(history.get("E", []))
    R = np.array(history.get("Lres", []))
    N = np.array(history.get("Lnorm", []))
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    axes[0].plot(E, label="E (RR)")
    axes[0].set_ylabel("Energy")
    axes[0].legend()

    axes[1].plot(R, label="Residual")
    axes[1].set_ylabel("PDE residual")
    axes[1].legend()

    axes[2].plot(N, label="Norm error")
    axes[2].set_ylabel("(<psi|psi>-1)^2")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
