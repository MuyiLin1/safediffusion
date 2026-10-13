"""
Visualise the robust violation set Omega_rob for OPEN 5 Item 2.

Usage:
    conda run -n dmwithllm python -m Lyapunov_guided_diffusion.sindy.plot_omega_rob \
        --csv data/unified_narrow.csv --sigmas 0.0 0.04 0.05 --output omega_rob.png
"""
from __future__ import annotations
import argparse, csv, os, tempfile
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

CLF_LAMBDA     = 1.0
THETA_RANGE    = (-2.0, 2.0)
THETADOT_RANGE = (-2.0, 2.0)
N_GRID = 50
N_CTRL = 20
CTRL_RANGE = (-10.0, 10.0)


def _build_grid():
    theta  = torch.linspace(*THETA_RANGE,    N_GRID)
    thetad = torch.linspace(*THETADOT_RANGE, N_GRID)
    mesh   = torch.meshgrid(theta, thetad, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1).float(), theta.numpy(), thetad.numpy()


def _h_rob_grid(controller, sindy_system, delta):
    x_grid, _, _ = _build_grid()
    x_v = x_grid.clone().requires_grad_(True)
    V = controller.V(x_v)
    grad_V = torch.autograd.grad(V.sum(), x_v, create_graph=False)[0]
    grad_norm = grad_V.norm(dim=-1)
    h_rob = torch.full((x_grid.shape[0],), float("inf"))
    for u_val in torch.linspace(*CTRL_RANGE, N_CTRL):
        u = torch.full((x_grid.shape[0], sindy_system.n_controls), float(u_val))
        with torch.no_grad():
            lie_V = (grad_V.detach() * sindy_system.closed_loop_dynamics(x_grid, u)).sum(-1)
        h_rob = torch.minimum(h_rob, lie_V + CLF_LAMBDA * V.detach().squeeze(-1) + grad_norm.detach() * delta)
    return h_rob.reshape(N_GRID, N_GRID).numpy()


def _train_for_viz(sindy_system, scenarios, max_epochs, accelerator):
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
    from neural_clbf.experiments import ExperimentSuite, CLFContourExperiment
    from neural_clbf.controllers import NeuralCLBFController
    dm = EpisodicDataModule(sindy_system, [(-1.5, 1.5), (-1.5, 1.5)],
        trajectories_per_episode=0, trajectory_length=1,
        fixed_samples=5000, max_points=50000, val_split=0.1, batch_size=64)
    ctrl = NeuralCLBFController(sindy_system, scenarios, dm,
        experiment_suite=ExperimentSuite([CLFContourExperiment(
            "V", domain=[(-2., 2.), (-2., 2.)], n_grid=15,
            x_axis_index=0, y_axis_index=1,
            x_axis_label=r"$\theta$", y_axis_label=r"$\dot\theta$",
            plot_unsafe_region=False)]),
        clbf_hidden_layers=2, clbf_hidden_size=64, clf_lambda=1.0,
        safe_level=1.0, controller_period=0.05, clf_relaxation_penalty=1e2,
        num_init_epochs=5, epochs_per_episode=100, barrier=False, disable_gurobi=True)
    pl.Trainer(logger=pl_loggers.TensorBoardLogger(tempfile.mkdtemp(), name="r"),
        max_epochs=max_epochs, accelerator=accelerator,
        gradient_clip_val=10.0, enable_progress_bar=False, log_every_n_steps=50
    ).fit(ctrl)
    ctrl.eval()
    return ctrl


def _run_one_sigma(sigma, csv_path, accelerator="cpu"):
    from neural_clbf.systems import InvertedPendulum
    from neural_clbf.systems.sindy_estimated_system import SINDyEstimatedSystem
    from Lyapunov_guided_diffusion.sindy.collect_data import collect_safe_transitions
    from Lyapunov_guided_diffusion.sindy.identify_dynamics import identify_dynamics
    row = None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if abs(float(r["obs_noise_std"]) - sigma) < 1e-6 and int(r["seed"]) == 0:
                row = r; break
    if row is None:
        print(f"  No row for sigma={sigma}"); return None
    params = {"m": 1.0, "L": 1.0, "b": 0.01}; scenarios = [params]
    torch.manual_seed(0); np.random.seed(0)
    true_sys = InvertedPendulum(params, dt=0.01, controller_dt=0.05, scenarios=scenarios)
    X, U, Xd = collect_safe_transitions(true_sys,
        num_trajectories=int(row["num_trajectories"]),
        trajectory_length=int(row["trajectory_length"]),
        exploration_std=0.5, obs_noise_std=sigma, seed=0)
    model = identify_dynamics(X, U, Xd, poly_degree=1)
    sindy_sys = SINDyEstimatedSystem(model, reference_system=true_sys, dt=0.01, controller_dt=0.05, scenarios=scenarios)
    ctrl = _train_for_viz(sindy_sys, scenarios, int(row["max_epochs"]), accelerator)
    return ctrl, sindy_sys, float(row["delta_sup"])


def make_figure(csv_path, sigmas, out_path, accelerator="cpu"):
    theta_t  = np.linspace(*THETA_RANGE,    N_GRID)
    thetad_t = np.linspace(*THETADOT_RANGE, N_GRID)
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5*len(sigmas), 4.8), sharex=True, sharey=True)
    if len(sigmas) == 1: axes = [axes]
    for ax, sigma in zip(axes, sigmas):
        print(f"Computing Omega_rob for sigma={sigma} ...")
        res = _run_one_sigma(sigma, csv_path, accelerator)
        if res is None: ax.set_title(f"sigma={sigma} (no data)"); continue
        ctrl, sindy_sys, delta = res
        h_rob = _h_rob_grid(ctrl, sindy_sys, delta)
        print(f"  delta={delta:.2f}, Omega_rob={float((h_rob>0).mean()):.3f}")
        ax.imshow((h_rob > 0).T, origin="lower", extent=[*THETA_RANGE, *THETADOT_RANGE],
            aspect="auto", cmap=matplotlib.colors.ListedColormap(["#c6dcf7","#f7c6c6"]), vmin=0, vmax=1)
        ax.contour(theta_t, thetad_t, h_rob.T, levels=[0], colors="black", linewidths=1.2)
        ax.set_title(rf"$\sigma={sigma}$, $\delta={delta:.1f}$"+"\n"+r"$\Omega_{\rm rob}$ in red", fontsize=10)
        ax.set_xlabel(r"$\theta$")
        if ax is axes[0]: ax.set_ylabel(r"$\dot\theta$")
    fig.legend(handles=[mpatches.Patch(color="#c6dcf7", label=r"certified"),
        mpatches.Patch(color="#f7c6c6", label=r"$\Omega_{\rm rob}$")],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.04))
    fig.tight_layout(rect=(0,0.06,1,1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default=os.path.join(_REPO,"data","unified_narrow.csv"))
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.04, 0.05])
    ap.add_argument("--output", default=os.path.join(_REPO,"omega_rob.png"))
    ap.add_argument("--accelerator", default="cpu")
    args = ap.parse_args()
    make_figure(args.csv, args.sigmas, args.output, args.accelerator)

if __name__ == "__main__":
    main()
