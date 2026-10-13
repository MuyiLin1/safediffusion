"""
Delta-scope experiment (resolves the open issue flagged in paper Section 5.4).

Section 5.3 measures the model error delta over a fixed operating box [-1,1]^2.
Section 5.4 then shows the boundary-aware ("wide") models are *safer* yet carry a
*larger* box-delta than the interior ("narrow") models -- so a box-restricted
delta is not a cross-distribution safety predictor. The reason is geometric: the
unsafe overshoots happen on the *approach* to the boundary (||x|| in [1.0, 1.5]),
which lies just OUTSIDE the [-1,1]^2 box the delta is measured over. Theorem 12
(finite-horizon probabilistic safety) already localizes its delta to the rollout
tube, not the certified box.

This script recomputes delta over regions that actually contain the safety-
critical states, for BOTH data-collection regimes, without retraining anything a
second time:

  Exp 1 (region ablation): delta over a family of growing boxes
        [-s,s]^2 for s in {1.0, 1.25, 1.5, 1.75, 2.0}. Expectation: the narrow
        model's delta jumps as s grows (it extrapolates poorly on the approach),
        while the wide model's barely moves -- so the ordering flips to match
        safety.

  Exp 2 (reachable-set delta): delta over the states the TRUE closed loop
        actually visits (pre-divergence), i.e. delta_reach. Expectation: safety
        becomes (near-)monotone in delta_reach ACROSS both regimes, where it was
        not in delta_box. Pool all rows and plot safety vs delta_reach to get a
        single collapsing curve.

This is a NEW file: it reuses sweep.py's primitives (data collection, SINDy
identification, CLBF training, true-system rollout) but writes its own results
CSV and never mutates sweep.py or its outputs, so the Section 5.3/5.4 code and
this follow-up remain separate, submission-ready artifacts.

Usage (cluster, full):
    CUDA_VISIBLE_DEVICES=-1 python -m Lyapunov_guided_diffusion.sindy.sweep_delta_scope \
        --ic_radius 0.0 1.4 --obs_noise_std 0.04 0.05 0.075 0.10 0.15 \
        --seeds 0 1 2 3 4 --num_trajectories 50 --max_epochs 5 \
        --num_rollouts 30 --t_sim 10 --output /tmp/delta_scope.csv

Usage (tiny pilot):
    CUDA_VISIBLE_DEVICES=-1 python -m Lyapunov_guided_diffusion.sindy.sweep_delta_scope \
        --ic_radius 0.0 1.4 --obs_noise_std 0.05 --seeds 0 \
        --num_trajectories 50 --max_epochs 5 --num_rollouts 30 --t_sim 10 \
        --output /tmp/delta_scope_pilot.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from neural_clbf.systems import InvertedPendulum
from neural_clbf.systems.sindy_estimated_system import SINDyEstimatedSystem

from Lyapunov_guided_diffusion.sindy.collect_data import collect_safe_transitions
from Lyapunov_guided_diffusion.sindy.identify_dynamics import identify_dynamics
from Lyapunov_guided_diffusion.sindy.evaluate import evaluate_on_true_system
from Lyapunov_guided_diffusion.sindy.sweep import steady_state_V
from Lyapunov_guided_diffusion.training.train_inverted_pendulum_sindy import build_and_train


# Growing evaluation boxes for the region ablation (Exp 1). The cliff / unsafe
# boundary is at ||x|| = 1.5; the state limits are the +-2 box. box_1.0 is the
# region Section 5.3 used.
REGION_BOXES = {
    "box_1.0": ((-1.0, 1.0), (-1.0, 1.0)),
    "box_1.25": ((-1.25, 1.25), (-1.25, 1.25)),
    "box_1.5": ((-1.5, 1.5), (-1.5, 1.5)),
    "box_1.75": ((-1.75, 1.75), (-1.75, 1.75)),
    "box_2.0": ((-2.0, 2.0), (-2.0, 2.0)),
}

SENTINEL = 2.0  # evaluate.py freezes diverged rollouts at this out-of-bounds value


def _delta_stats(sindy_system, true_system, x, u_range, gen):
    """sup / mean / median ||f - f_hat|| at states x, u ~ Uniform(u_range)."""
    u_lo, u_hi = float(u_range[0]), float(u_range[1])
    u = (
        torch.rand(x.shape[0], true_system.n_controls, generator=gen)
        * (u_hi - u_lo)
        + u_lo
    )
    with torch.no_grad():
        xdot_true = true_system.closed_loop_dynamics(x, u)
        xdot_est = sindy_system.closed_loop_dynamics(x, u)
    err = torch.norm(xdot_true - xdot_est, dim=-1)
    return {
        "sup": float(err.max().item()),
        "mean": float(err.mean().item()),
        "median": float(err.median().item()),
    }


def delta_over_box(sindy_system, true_system, box, u_range, seed, n_grid=60):
    """delta over a gridded box (same construction as sweep.measure_delta)."""
    gen = torch.Generator().manual_seed(seed)
    axes = [torch.linspace(lo, hi, n_grid) for (lo, hi) in box]
    mesh = torch.meshgrid(*axes, indexing="ij")
    x = torch.stack([m.reshape(-1) for m in mesh], dim=-1).float()
    return _delta_stats(sindy_system, true_system, x, u_range, gen)


def delta_over_states(sindy_system, true_system, states, u_range, seed):
    """delta over an explicit point cloud of states (the reachable set)."""
    if states.size == 0:
        return {"sup": float("nan"), "mean": float("nan"), "median": float("nan")}
    gen = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(states, dtype=torch.float32)
    return _delta_stats(sindy_system, true_system, x, u_range, gen)


def reachable_states(results):
    """States the TRUE closed loop actually visits, before any divergence.

    Diverged rollouts are frozen at the SENTINEL corner by evaluate.py; we keep
    only the genuine pre-divergence states (indices < exit_step) and drop any
    residual at/beyond the sentinel so the reachable set reflects real transient
    excursions toward the boundary, not the artificial freeze.
    """
    traj = results["trajectories"]  # (R, T+1, n)
    exit_step = results["exit_step"]  # (R,), -1 = survived
    R, Tp1, _ = traj.shape
    chunks = []
    for i in range(R):
        end = Tp1 if int(exit_step[i]) < 0 else int(exit_step[i])
        if end > 0:
            chunks.append(traj[i, :end, :])
    if not chunks:
        return np.empty((0, traj.shape[-1]), dtype=np.float32)
    X = np.concatenate(chunks, axis=0)
    keep = np.max(np.abs(X), axis=1) < (SENTINEL - 1e-6)
    return X[keep]


def run_one_scope(
    obs_noise_std: float,
    seed: int,
    ic_radius: float,
    num_trajectories: int,
    trajectory_length: int,
    max_epochs: int,
    accelerator: str,
    num_rollouts: int,
    t_sim: float,
    poly_degree: int = 1,
):
    """Plain-SINDy pipeline + multi-region and reachable-set delta. One row."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    simulation_dt = 0.01
    controller_period = 0.05
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    scenarios = [nominal_params]

    true_system = InvertedPendulum(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # 1) Collect data (narrow ic_radius=0 or wide ic_radius>0; seed-deterministic).
    X, U, Xdot = collect_safe_transitions(
        true_system,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        exploration_std=0.5,
        obs_noise_std=obs_noise_std,
        seed=seed,
        ic_radius=ic_radius,
    )
    N = X.shape[0]
    u_range = (float(U.min()), float(U.max()))

    # 2-4) Plain SINDy (the winning lever from Section 5.4) + CLBF against f_hat.
    model = identify_dynamics(X, U, Xdot, poly_degree=poly_degree)
    sindy_system = SINDyEstimatedSystem(
        model, reference_system=true_system,
        dt=simulation_dt, controller_dt=controller_period, scenarios=scenarios,
    )
    controller = build_and_train(
        sindy_system, scenarios, max_epochs=max_epochs,
        accelerator=accelerator, enable_logger=False,
    )
    controller.eval()

    # 5) Roll out on the TRUE system; capture visited states.
    results = evaluate_on_true_system(
        controller, true_system,
        num_rollouts=num_rollouts, t_sim=t_sim,
        controller_period=controller_period,
    )
    safety_rate = float(results["safe_flags"].mean())
    goal_rate = float(results["goal_reached"].mean())
    limsup_V = steady_state_V(results["V_values"])

    reach = reachable_states(results)
    reach_norms = np.linalg.norm(reach, axis=1) if reach.size else np.array([])

    row = {
        "ic_radius": ic_radius,
        "obs_noise_std": obs_noise_std,
        "seed": seed,
        "N": N,
        "safety_rate": safety_rate,
        "goal_rate": goal_rate,
        "limsup_V": limsup_V,
        "u_lo": u_range[0],
        "u_hi": u_range[1],
        "n_reach": int(reach.shape[0]),
        "reach_max_norm": float(reach_norms.max()) if reach_norms.size else float("nan"),
        "reach_frac_gt1": float(np.mean(reach_norms > 1.0)) if reach_norms.size else float("nan"),
    }

    # Exp 1: delta over the family of growing boxes.
    for name, box in REGION_BOXES.items():
        d = delta_over_box(sindy_system, true_system, box, u_range, seed)
        row[f"delta_sup_{name}"] = d["sup"]
        row[f"delta_mean_{name}"] = d["mean"]

    # Exp 2: delta over the true reachable set.
    dr = delta_over_states(sindy_system, true_system, reach, u_range, seed)
    row["delta_sup_reach"] = dr["sup"]
    row["delta_mean_reach"] = dr["mean"]

    return row


def build_fieldnames():
    names = [
        "ic_radius", "obs_noise_std", "seed", "N",
        "safety_rate", "goal_rate", "limsup_V",
        "u_lo", "u_hi", "n_reach", "reach_max_norm", "reach_frac_gt1",
    ]
    for name in REGION_BOXES:
        names += [f"delta_sup_{name}", f"delta_mean_{name}"]
    names += ["delta_sup_reach", "delta_mean_reach"]
    return names


def main():
    parser = argparse.ArgumentParser(description="SINDy delta-scope experiment")
    parser.add_argument("--ic_radius", type=float, nargs="+", default=[0.0, 1.4],
                        help="Data-collection radii to compare (0=narrow, 1.4=wide)")
    parser.add_argument("--obs_noise_std", type=float, nargs="+",
                        default=[0.04, 0.05, 0.075, 0.10, 0.15])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_trajectories", type=int, default=50)
    parser.add_argument("--trajectory_length", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--poly_degree", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--num_rollouts", type=int, default=30)
    parser.add_argument("--t_sim", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="data/delta_scope.csv")
    args = parser.parse_args()

    fieldnames = build_fieldnames()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    f = open(args.output, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        f.flush()

    configs = [
        (icr, ns, seed)
        for icr in args.ic_radius
        for ns in args.obs_noise_std
        for seed in args.seeds
    ]
    print(f"Running {len(configs)} configurations -> {args.output}")

    for i, (icr, ns, seed) in enumerate(configs, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(configs)}] ic_radius={icr} noise={ns} seed={seed} ...")
        try:
            row = run_one_scope(
                obs_noise_std=ns,
                seed=seed,
                ic_radius=icr,
                num_trajectories=args.num_trajectories,
                trajectory_length=args.trajectory_length,
                max_epochs=args.max_epochs,
                accelerator=args.accelerator,
                num_rollouts=args.num_rollouts,
                t_sim=args.t_sim,
                poly_degree=args.poly_degree,
            )
        except Exception as exc:  # keep the sweep alive on a bad config
            print(f"    FAILED: {exc!r}")
            continue
        writer.writerow(row)
        f.flush()
        dt = time.time() - t0
        print(
            f"    N={row['N']} safe={row['safety_rate']:.2f} "
            f"d_box1.0={row['delta_sup_box_1.0']:.1f} "
            f"d_box1.5={row['delta_sup_box_1.5']:.1f} "
            f"d_reach={row['delta_sup_reach']:.1f} "
            f"reach_max={row['reach_max_norm']:.2f} ({dt:.0f}s)"
        )

    f.close()
    print(f"\nDelta-scope sweep complete. Results in {args.output}")


if __name__ == "__main__":
    main()
