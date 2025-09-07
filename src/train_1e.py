from __future__ import annotations

import argparse
from pathlib import Path

import torch
# Use double precision for more accurate second derivatives
try:
    torch.set_default_dtype(torch.float64)
except Exception:
    pass
import json
from .viz.plotting import plot_training_curves, plot_potential, plot_wavefunction_density



from .physics.units import GaAs, pretty_summary
from .potentials.biquadratic import BiquadraticParams, from_targets, v_biq_factory
from .pinn.models import SIREN
from .pinn.losses import rayleigh_ritz_energy, pde_residual_loss, orthogonality_loss


# ---- Helper functions for training loop ----

def vectorized_biq_v(p: BiquadraticParams, xy: torch.Tensor) -> torch.Tensor:
    # xy: [N,2]
    x = xy[:, 0:1]
    y = xy[:, 1:1+1]
    a2 = p.a * p.a
    return p.c4 * (x * x - a2) ** 2 + p.c2y * (y * y) + p.delta * x


def sample_collocation(X: float, Y: float, n: int, device: torch.device) -> torch.Tensor:
    # Uniform sampling in the box [-X,X]x[-Y,Y]
    xy = torch.empty((n, 2), device=device).uniform_(0.0, 1.0)
    xy[:, 0] = (xy[:, 0] * 2.0 - 1.0) * X
    xy[:, 1] = (xy[:, 1] * 2.0 - 1.0) * Y
    return xy


def normalization_penalty(model: torch.nn.Module, xy_q: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    psi = model(xy_q)
    norm = torch.sum(w_q * psi * psi)
    return (norm - 1.0) ** 2


def compute_norm_value(model: torch.nn.Module, xy_q: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    """Compute the normalization value ∫|psi|^2 dΩ on the quadrature grid."""
    with torch.no_grad():
        psi = model(xy_q)
        return torch.sum(w_q * psi * psi)


def train_one_state(
    model: torch.nn.Module,
    p: BiquadraticParams,
    xy_q: torch.Tensor,
    w_q: torch.Tensor,
    X: float,
    Y: float,
    epochs: int = 3000,
    n_colloc: int = 4096,
    lr: float = 1e-3,
    lam_rr: float = 1.0,
    lam_pde: float = 1.0,
    lam_norm: float = 1.0,
    device: torch.device | None = None,
    lbfgs_iters: int = 200,
    lam_ortho: float = 0.0,
    ref_model: torch.nn.Module | None = None,
) -> dict:
    device = device or xy_q.device

    def vfun_batch(xy: torch.Tensor) -> torch.Tensor:
        return vectorized_biq_v(p, xy)

    if ref_model is not None:
        ref_model.eval()
        for p_ref in ref_model.parameters():
            p_ref.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"E": [], "Lres": [], "Lnorm": [], "NormValue": [], "Overlap0": []}
    best = {"E": float("inf"), "state_dict": None}

    for ep in range(1, epochs + 1):
        model.train()
        xy_c = sample_collocation(X, Y, n_colloc, device)

        E = rayleigh_ritz_energy(model, vfun_batch, xy_q, w_q)
        Lres = pde_residual_loss(model, vfun_batch, xy_c)
        Lnorm = normalization_penalty(model, xy_q, w_q)
        loss = lam_rr * E + lam_pde * Lres + lam_norm * Lnorm

        # Optional orthogonality to reference (ground state) model
        overlap_val = None
        if ref_model is not None and lam_ortho > 0.0:
            with torch.no_grad():
                psi_prev = ref_model(xy_q)
            psi_new = model(xy_q)
            overlap_sq = orthogonality_loss(psi_new, psi_prev, w_q)
            loss = loss + lam_ortho * overlap_sq
            overlap_val = torch.sqrt(overlap_sq + 1e-12)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        history["E"].append(float(E.detach().cpu()))
        history["Lres"].append(float(Lres.detach().cpu()))
        history["Lnorm"].append(float(Lnorm.detach().cpu()))
        # Track actual normalization value
        norm_val = compute_norm_value(model, xy_q, w_q)
        history["NormValue"].append(float(norm_val.detach().cpu()))
        if overlap_val is not None:
            history["Overlap0"].append(float(overlap_val.detach().cpu()))

        if E.item() < best["E"]:
            best["E"] = float(E.item())
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0 or ep == 1:
            msg = f"[ep {ep:5d}] E={E.item():.6f}  Lres={Lres.item():.3e}  Lnorm={Lnorm.item():.3e}  |psi|^2={norm_val.item():.5f}"
            if overlap_val is not None:
                msg += f"  overlap0={overlap_val.item():.3e}"
            print(msg)

    # Optional LBFGS refinement
    if lbfgs_iters and lbfgs_iters > 0:
        opt2 = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iters, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad(set_to_none=True)
            xy_c2 = sample_collocation(X, Y, max(256, n_colloc // 4), device)
            E2 = rayleigh_ritz_energy(model, vfun_batch, xy_q, w_q)
            R2 = pde_residual_loss(model, vfun_batch, xy_c2)
            N2 = normalization_penalty(model, xy_q, w_q)
            loss2 = lam_rr * E2 + lam_pde * R2 + lam_norm * N2
            if ref_model is not None and lam_ortho > 0.0:
                with torch.no_grad():
                    psi_prev2 = ref_model(xy_q)
                psi_new2 = model(xy_q)
                overlap_sq2 = orthogonality_loss(psi_new2, psi_prev2, w_q)
                loss2 = loss2 + lam_ortho * overlap_sq2
            loss2.backward()
            return loss2

        opt2.step(closure)
        # One last log after LBFGS and update best if improved
        with torch.no_grad():
            E2 = rayleigh_ritz_energy(model, vfun_batch, xy_q, w_q)
            R2 = pde_residual_loss(model, vfun_batch, xy_q[::10])
            N2 = normalization_penalty(model, xy_q, w_q)
            norm_val2 = compute_norm_value(model, xy_q, w_q)
            e2f = float(E2.detach().cpu())
            history["E"].append(e2f)
            history["Lres"].append(float(R2.detach().cpu()))
            history["Lnorm"].append(float(N2.detach().cpu()))
            history["NormValue"].append(float(norm_val2.detach().cpu()))
            if ref_model is not None and lam_ortho > 0.0:
                psi_prev_f = ref_model(xy_q)
                psi_new_f = model(xy_q)
                overlap_sq_f = orthogonality_loss(psi_new_f, psi_prev_f, w_q)
                history["Overlap0"].append(float(torch.sqrt(overlap_sq_f + 1e-12).detach().cpu()))
            if e2f < best["E"]:
                best["E"] = e2f
                best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Load best (across Adam+LBFGS)
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    return {"history": history, "bestE": best["E"]}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train 1e PINN for DQD biquadratic potential (Rayleigh–Ritz + residual)")
    # Potential parameters
    ap.add_argument("--a", type=float, default=1.5)
    ap.add_argument("--c4", type=float, default=None)
    ap.add_argument("--c2y", type=float, default=None)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--hbar-omega-x", type=float, default=3.0, help="meV; used if c4 not provided")
    ap.add_argument("--hbar-omega-y", type=float, default=5.0, help="meV; used if c2y not provided")
    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nq", type=int, default=96, help="quadrature grid per axis")
    ap.add_argument("--nc", type=int, default=4096, help="number of collocation points per epoch")
    ap.add_argument("--lam-rr", type=float, default=1.0)
    ap.add_argument("--lam-pde", type=float, default=1.0)
    ap.add_argument("--lam-norm", type=float, default=1.0)
    ap.add_argument("--lbfgs-iters", type=int, default=200, help="LBFGS refinement iterations (0 to disable)")
    # Excited state / orthogonality options
    ap.add_argument("--state", type=int, default=0, help="0 = ground state, 1 = first excited (orthogonal to ref)")
    ap.add_argument("--ref-model", type=str, default=None, help="Path to reference model (model_best.pt) for orthogonality constraint")
    ap.add_argument("--lam-ortho", type=float, default=10.0, help="Weight for orthogonality penalty when state>0")
    # System
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, default="data/run_1e")
    ap.add_argument("--grid-save", type=int, default=200, help="save psi on an N x N grid for plots")
    ap.add_argument("--no-plots", action="store_true", help="do not save plots")
    return ap.parse_args()


def make_quadrature_box(X: float, Y: float, n: int = 128):
    xs = torch.linspace(-X, X, n)
    ys = torch.linspace(-Y, Y, n)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=-1)
    w = torch.full((xy.shape[0], 1), (2*X / (n-1)) * (2*Y / (n-1)))
    return xy, w



def compute_density_features(xs_t: torch.Tensor, ys_t: torch.Tensor, psi_grid_t: torch.Tensor, a_param: float | None = None) -> dict:
    """Compute simple diagnostics on |psi|^2: COM, side masses, top-2 peaks.
    xs_t, ys_t: 1D tensors of length N; psi_grid_t: [N,N] tensor.
    """
    N = xs_t.numel()
    P = (psi_grid_t ** 2)  # [N,N]
    # Normalize using trapezoid-like cell area (uniform grid):
    dx = (xs_t[-1] - xs_t[0]) / (N - 1)
    dy = (ys_t[-1] - ys_t[0]) / (N - 1)
    cell = dx * dy
    total_prob = float((P.sum() * cell).item())

    Xg = xs_t.view(N, 1).expand(N, N)
    Yg = ys_t.view(1, N).expand(N, N)

    com_x = float(((P * Xg).sum() * cell / (total_prob + 1e-12)).item())
    com_y = float(((P * Yg).sum() * cell / (total_prob + 1e-12)).item())

    left_mask = (Xg < 0)
    right_mask = ~left_mask
    left_mass = float((P[left_mask].sum() * cell).item())
    right_mass = float((P[right_mask].sum() * cell).item())

    # Find top-2 peaks (global maxima) and their coordinates
    P_flat = P.reshape(-1)
    vals, idxs = torch.topk(P_flat, k=min(2, P_flat.numel()))
    peaks = []
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        iy = idx % N
        ix = idx // N
        peaks.append({"x": float(xs_t[ix].item()), "y": float(ys_t[iy].item()), "p": float(v)})

    # Optional: tie to expected minima at (±a,0) and sample density there
    minima = None
    minima_density = None
    if a_param is not None:
        minima = [{"x": -float(a_param), "y": 0.0}, {"x": float(a_param), "y": 0.0}]
        # nearest grid indices to (-a,0) and (+a,0)
        ixL = int(torch.argmin(torch.abs(xs_t - (-float(a_param))))).item()
        ixR = int(torch.argmin(torch.abs(xs_t - (float(a_param))))).item()
        iy0 = int(torch.argmin(torch.abs(ys_t - 0.0))).item()
        pL = float(P[iy0, ixL].item())
        pR = float(P[iy0, ixR].item())
        minima_density = [
            {"x": float(xs_t[ixL].item()), "y": float(ys_t[iy0].item()), "p": pL},
            {"x": float(xs_t[ixR].item()), "y": float(ys_t[iy0].item()), "p": pR},
        ]

    return {
        "grid_N": int(N),
        "dx": float(dx.item()),
        "dy": float(dy.item()),
        "total_prob": total_prob,
        "com": {"x": com_x, "y": com_y},
        "side_mass": {"left": left_mass, "right": right_mass},
        "peaks": peaks,
        "expected_minima": minima,
        "minima_density": minima_density,
    }

def main():
    args = parse_args()
    mat = GaAs()
    print(pretty_summary(mat))

    if args.c4 is None or args.c2y is None:
        p = from_targets(a=args.a, hbar_omega_x_meV=args.hbar_omega_x, hbar_omega_y_meV=args.hbar_omega_y, delta=args.delta, mat=mat)
    else:
        p = BiquadraticParams(a=args.a, c4=float(args.c4), c2y=float(args.c2y), delta=args.delta)

    print(f"Using biquadratic params: a={p.a:.3f}, c4={p.c4:.3f}, c2y={p.c2y:.3f}, delta={p.delta:.3f}")

    X = Y = 3.0 * max(1.2, p.a) + 1.0
    vfun = v_biq_factory(p)

    device = torch.device(args.device)
    model = SIREN(in_features=2, hidden_features=args.hidden, hidden_layers=args.layers, out_features=1).to(device)

    # Quadrature grid for RR energy and normalization
    xy_q, w_q = make_quadrature_box(X, Y, n=args.nq)
    xy_q = xy_q.to(device)
    w_q = w_q.to(device)

    # Optional reference model for orthogonality (excited state)
    ref_model = None
    if getattr(args, "state", 0) > 0 and getattr(args, "ref_model", None):
        ref_model = SIREN(in_features=2, hidden_features=args.hidden, hidden_layers=args.layers, out_features=1).to(device)
        ref_sd = torch.load(args.ref_model, map_location=device)
        ref_model.load_state_dict(ref_sd)
        ref_model.eval()

    # Train state (ground if state=0, excited if state>0 with orthogonality)
    res = train_one_state(
        model,
        p,
        xy_q,
        w_q,
        X,
        Y,
        epochs=args.epochs,
        n_colloc=args.nc,
        lr=args.lr,
        lam_rr=args.lam_rr,
        lam_pde=args.lam_pde,
        lam_norm=args.lam_norm,
        device=device,
        lbfgs_iters=args.lbfgs_iters,
        lam_ortho=(args.lam_ortho if ref_model is not None else 0.0),
        ref_model=ref_model,
    )

    # Evaluate final metrics (autograd needed for Laplacian)
    vfun_batch = lambda xy: vectorized_biq_v(p, xy)
    E_fin = rayleigh_ritz_energy(model, vfun_batch, xy_q, w_q)
    Lres_fin = pde_residual_loss(model, vfun_batch, xy_q[:: max(1, (args.nq*args.nq)//2000) ])
    print(f"Final: E={E_fin.item():.6f}, residual={Lres_fin.item():.3e}, bestE={res['bestE']:.6f}")

    # Save outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "params.txt").write_text(pretty_summary(mat) + "\n" + f"{p}\n", encoding="utf-8")

    # Final metrics (include residual and normalization)
    Lnorm_fin = normalization_penalty(model, xy_q, w_q)
    norm_val_fin = compute_norm_value(model, xy_q, w_q)
    (outdir / "energies.txt").write_text(
        f"E_final={E_fin.item():.8f}\n"
        f"residual_final={Lres_fin.item():.8e}\n"
        f"norm_value_final={norm_val_fin.item():.8f}\n"
        f"norm_penalty_final={Lnorm_fin.item():.8e}\n"
        f"E_best={res['bestE']:.8f}\n",
        encoding="utf-8",
    )

    # Save training history (JSON + CSV)
    hist = res.get("history", {})
    (outdir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    with open(outdir / "history.csv", "w", encoding="utf-8") as f:
        f.write("epoch,E,Lres,Lnorm,NormValue,Overlap0\n")
        E_list = hist.get("E", [])
        R_list = hist.get("Lres", [])
        N_list = hist.get("Lnorm", [])
        NV_list = hist.get("NormValue", [])
        OV_list = hist.get("Overlap0", [])
        for i in range(len(E_list)):
            f.write(
                f"{i+1},{E_list[i]}"
                f",{R_list[i] if i < len(R_list) else ''}"
                f",{N_list[i] if i < len(N_list) else ''}"
                f",{NV_list[i] if i < len(NV_list) else ''}"
                f",{OV_list[i] if i < len(OV_list) else ''}\n"
            )

    # Save config/args
    cfg = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "params": {"a": p.a, "c4": p.c4, "c2y": p.c2y, "delta": p.delta},
        "box": {"X": float(X), "Y": float(Y), "nq": int(args.nq), "nc": int(args.nc)},
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Save best model weights
    torch.save(model.state_dict(), outdir / "model_best.pt")

    # Save psi on a grid for plotting and analysis
    N = int(args.grid_save)
    xs = torch.linspace(-X, X, N, device=device)
    ys = torch.linspace(-Y, Y, N, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    xy_eval = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=-1)
    with torch.no_grad():
        psi_grid = model(xy_eval).reshape(N, N).cpu()

    # Density diagnostics (peaks, COM, side masses)
    feats = compute_density_features(xs.cpu(), ys.cpu(), psi_grid, a_param=p.a)
    (outdir / "density_features.json").write_text(json.dumps(feats, indent=2), encoding="utf-8")
    # Minimal CSV for easy glance
    with open(outdir / "density_features.csv", "w", encoding="utf-8") as fdf:
        fdf.write("total_prob,com_x,com_y,left_mass,right_mass,peak1_x,peak1_y,peak1_p,peak2_x,peak2_y,peak2_p\n")
        pk1 = feats["peaks"][0] if len(feats.get("peaks", []))>0 else {"x":"","y":"","p":""}
        pk2 = feats["peaks"][1] if len(feats.get("peaks", []))>1 else {"x":"","y":"","p":""}
        fdf.write(
            f"{feats['total_prob']},{feats['com']['x']},{feats['com']['y']},{feats['side_mass']['left']},{feats['side_mass']['right']},"
            f"{pk1.get('x','')},{pk1.get('y','')},{pk1.get('p','')},{pk2.get('x','')},{pk2.get('y','')},{pk2.get('p','')}\n"
        )

    # Plots
    if not args.no_plots:
        plot_training_curves(res["history"], out_path=str(outdir / "training_curves.png"))
        figV = plot_potential(lambda x, y: vfun(float(x), float(y)), X, Y, nx=200, ny=200, title="Potential v(x,y)")
        figV.savefig(outdir / "potential.png", dpi=150)
        xs_np = xs.cpu().numpy(); ys_np = ys.cpu().numpy(); psi_np = psi_grid.numpy()
        figP = plot_wavefunction_density(
            xs_np,
            ys_np,
            psi_np.reshape(-1),
            N,
            N,
            title=r"$|\psi|^2$",
            cmap="inferno",
            gamma=0.6,
            overlay_contours=True,
            contour_levels=12,
            overlay_points=[(-p.a, 0.0), (p.a, 0.0)],
            aspect="equal",
        )
        figP.savefig(outdir / "psi_density.png", dpi=150)

    torch.save({"x": xs.cpu(), "y": ys.cpu(), "psi": psi_grid}, outdir / "psi_grid.pt")


if __name__ == "__main__":
    main()

