"""
Data trade-off sweep: how much does learned-dynamics safety degrade as the
SINDy model error delta grows?

For each configuration (dataset size x measurement-noise x seed) this script:
  1. collects transition data from the true system,
  2. identifies f_hat with SINDy,
  3. measures the model error delta over the operating region,
  4. trains the CLBF certificate entirely against f_hat,
  5. rolls the resulting controller out on the TRUE system and records
     downstream safety (safe-set exit rate, steady-state V), and
  6. estimates the Theorem 6 constants (B, lambda_1, C eps^{1/n}).

Each run appends one row to data/sweep_results.csv. The companion script
plot_tradeoff.py turns that table into the headline figure: measured safety
degradation vs delta, with the predicted bound C eps^{1/n} + B delta / lambda_1
overlaid.

Usage (small local pilot):
    python -m Lyapunov_guided_diffusion.sindy.sweep \
        --num_trajectories 5 25 200 --obs_noise_std 0.0 \
        --seeds 0 --max_epochs 20 --accelerator cpu

Usage (full cluster sweep, GPU):
    python -m Lyapunov_guided_diffusion.sindy.sweep \
        --num_trajectories 2 5 10 25 50 200 \
        --obs_noise_std 0.0 0.05 0.1 0.2 \
        --seeds 0 1 2 --max_epochs 30 --accelerator gpu
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
from Lyapunov_guided_diffusion.sindy.theory_constants import (
    estimate_B,
    estimate_lambda1,
    estimate_Ceps,
)
from Lyapunov_guided_diffusion.training.train_inverted_pendulum_sindy import build_and_train


# Operating region used for the reported delta (matches the paper's setup).
OPERATING_DOMAIN = ((-1.0, 1.0), (-1.0, 1.0))
CLF_LAMBDA = 1.0


def measure_delta(sindy_system, true_system, n_grid: int = 60, seed: int = 0,
                  u_range=None):
    """Worst-case / mean / median ||f - f_hat|| over the operating region.

    Restricted to the operating box (not the full state space) so the value
    matches the delta the certificate and sampler actually experience.

    u_range: (lo, hi) interval of control values over which to evaluate the
    model error. This MUST be the range the closed-loop system actually
    commands, not true_system.control_limits: the QP box here is +-1000, which
    the controller never reaches, so probing delta there multiplies any tiny
    control-coefficient error by ~1000 and turns a benign model (true delta of a
    few units) into an apparent catastrophe (delta of hundreds). When u_range is
    None we fall back to control_limits; the sweep always passes the realized
    data range.
    """
    g = torch.Generator().manual_seed(seed)
    axes = [torch.linspace(lo, hi, n_grid) for (lo, hi) in OPERATING_DOMAIN]
    mesh = torch.meshgrid(*axes, indexing="ij")
    x = torch.stack([m.reshape(-1) for m in mesh], dim=-1).float()
    if u_range is not None:
        u_lo, u_hi = float(u_range[0]), float(u_range[1])
    else:
        u_upper, u_lower = true_system.control_limits
        u_lo, u_hi = float(u_lower), float(u_upper)
    u = (
        torch.rand(x.shape[0], true_system.n_controls, generator=g)
        * (u_hi - u_lo)
        + u_lo
    )
    with torch.no_grad():
        xdot_true = true_system.closed_loop_dynamics(x, u)
        xdot_est = sindy_system.closed_loop_dynamics(x, u)
    err = torch.norm(xdot_true - xdot_est, dim=-1)
    return {
        "delta_sup": float(err.max().item()),
        "delta_mean": float(err.mean().item()),
        "delta_median": float(err.median().item()),
    }


def steady_state_V(V_values: np.ndarray, tail_frac: float = 0.2) -> float:
    """Mean certificate value over the final tail of ALL rollouts."""
    T = V_values.shape[1]
    k = max(1, int(tail_frac * T))
    return float(V_values[:, -k:].mean())


def steady_state_V_survivors(
    V_values: np.ndarray, diverged: np.ndarray, tail_frac: float = 0.2
) -> float:
    """Mean tail certificate value over only the SURVIVING rollouts.

    Option B: a diverged trajectory is permanently lost (the unstable pendulum
    never recovers once it runs away), so averaging it into limsup_V would just
    measure the sentinel. Restricting to survivors makes limsup_V a clean test
    of the bound; the companion exit_rate/n_survivors columns report how many
    diverged so this number is never read in isolation (survivorship bias).
    Returns NaN when nothing survived.
    """
    survivors = ~diverged
    if not survivors.any():
        return float("nan")
    T = V_values.shape[1]
    k = max(1, int(tail_frac * T))
    return float(V_values[survivors, -k:].mean())


def run_one(
    num_trajectories: int,
    trajectory_length: int,
    obs_noise_std: float,
    seed: int,
    max_epochs: int,
    accelerator: str,
    num_rollouts: int,
    t_sim: float,
    poly_degree: int = 1,
    safety_aware: bool = False,
    refine_rounds: int = 1,
    refine_rho: float = 1.0,
    refine_objective: str = "volume",
    ic_radius: float = 0.0,
    robust_delta: float = 0.0,
):
    """Execute one full pipeline run and return a results row (dict)."""
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

    # 1) Collect (deliberately degraded) data.
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
    refine_diag = {}

    if safety_aware:
        # 2-4) Safety-aware alternating loop (Item #3): warm-start SINDy ->
        #      train CLBF -> refine f_hat to shrink Omega_rob -> retrain, for
        #      refine_rounds. refine_rounds=0 is the torch warm-start baseline
        #      (same library, no volume penalty) -- the clean control.
        from Lyapunov_guided_diffusion.sindy.safety_aware_sindy import (
            alternating_refine,
        )

        def _train(sys, sc):
            return build_and_train(
                sys, sc, max_epochs=max_epochs,
                accelerator=accelerator, enable_logger=False,
            )

        out = alternating_refine(
            X, U, Xdot,
            reference_system=true_system,
            train_clbf_fn=_train,
            scenarios=scenarios,
            n_rounds=refine_rounds,
            lam=CLF_LAMBDA,
            rho=refine_rho,
            objective=refine_objective,
            u_range=u_range,
            dt=simulation_dt,
            controller_dt=controller_period,
            verbose=True,
        )
        model = out["model"]
        controller = out["controller"]
        if out["history"]:
            refine_diag = out["history"][-1]
        sindy_system = SINDyEstimatedSystem(
            model, reference_system=true_system,
            dt=simulation_dt, controller_dt=controller_period, scenarios=scenarios,
        )
    else:
        # 2) Identify f_hat (plain SINDy). poly_degree=1 keeps only the terms
        #    the true pendulum needs; higher degrees overfit the noise.
        model = identify_dynamics(X, U, Xdot, poly_degree=poly_degree)
        # 3) Wrap.
        sindy_system = SINDyEstimatedSystem(
            model, reference_system=true_system,
            dt=simulation_dt, controller_dt=controller_period, scenarios=scenarios,
        )
        # 4) Train the certificate against f_hat (no TB logging during sweeps).
        controller = build_and_train(
            sindy_system, scenarios, max_epochs=max_epochs,
            accelerator=accelerator, enable_logger=False,
            robust_delta=robust_delta,
        )

    controller.eval()
    # Delta over the realized control range (see measure_delta docstring), NOT
    # the +-1000 QP control_limits which would inflate delta ~1000x.
    delta = measure_delta(sindy_system, true_system, seed=seed, u_range=u_range)

    # 5) Roll out on the TRUE system: downstream safety.
    results = evaluate_on_true_system(
        controller,
        true_system,
        num_rollouts=num_rollouts,
        t_sim=t_sim,
        controller_period=controller_period,
    )
    safety_rate = float(results["safe_flags"].mean())
    goal_rate = float(results["goal_reached"].mean())
    limsup_V = steady_state_V(results["V_values"])
    t_array = np.linspace(0, t_sim, results["V_values"].shape[1])

    # Option B survivor statistics.
    diverged = results["diverged"]
    n_survivors = int((~diverged).sum())
    limsup_V_survivors = steady_state_V_survivors(results["V_values"], diverged)
    exit_steps = results["exit_step"][diverged]
    mean_exit_time = (
        float(exit_steps.mean() * controller_period) if exit_steps.size else float("nan")
    )

    # 6) Theory constants and predicted bound.
    B = estimate_B(controller, domain=OPERATING_DOMAIN)
    lambda1 = estimate_lambda1(results["V_values"], t_array)
    ceps = estimate_Ceps(
        controller, sindy_system, clf_lambda=CLF_LAMBDA, domain=OPERATING_DOMAIN
    )
    predicted_limsup = ceps["C_eps_term"] + B * delta["delta_sup"] / lambda1

    return {
        "num_trajectories": num_trajectories,
        "trajectory_length": trajectory_length,
        "N": N,
        "obs_noise_std": obs_noise_std,
        "seed": seed,
        "max_epochs": max_epochs,
        "poly_degree": poly_degree,
        "delta_sup": delta["delta_sup"],
        "delta_mean": delta["delta_mean"],
        "delta_median": delta["delta_median"],
        "safety_rate": safety_rate,
        "exit_rate": 1.0 - safety_rate,
        "goal_rate": goal_rate,
        "limsup_V": limsup_V,
        "limsup_V_survivors": limsup_V_survivors,
        "n_survivors": n_survivors,
        "num_rollouts": num_rollouts,
        "mean_exit_time": mean_exit_time,
        "B": B,
        "lambda1": lambda1,
        "bad_set_fraction": ceps["bad_set_fraction"],
        "C_eps_term": ceps["C_eps_term"],
        "predicted_limsup": predicted_limsup,
        "safety_aware": int(safety_aware),
        "refine_rounds": refine_rounds if safety_aware else 0,
        "refine_vol_before": refine_diag.get("vol_surrogate_before", float("nan")),
        "refine_vol_after": refine_diag.get("vol_surrogate_after", float("nan")),
        "refine_data_ratio": refine_diag.get("data_mse_ratio", float("nan")),
        "refine_cheated": int(refine_diag.get("cheated", 0)),
        "refine_objective": refine_objective if safety_aware else "",
    }


def main():
    parser = argparse.ArgumentParser(description="SINDy data trade-off sweep")
    parser.add_argument("--num_trajectories", type=int, nargs="+",
                        default=[2, 5, 10, 25, 50, 200],
                        help="Dataset-size knob (trajectories per config)")
    parser.add_argument("--trajectory_length", type=int, default=100)
    parser.add_argument("--obs_noise_std", type=float, nargs="+",
                        default=[0.0],
                        help="Measurement-noise knob (one run per value)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--poly_degree", type=int, default=1,
                        help="SINDy polynomial library degree "
                             "(1=lean/principled, 3=rich/overfits noise)")
    parser.add_argument("--safety_aware", action="store_true",
                        help="Use the volume-penalized alternating SINDy loop "
                             "(Item #3) instead of plain SINDy")
    parser.add_argument("--refine_rounds", type=int, default=1,
                        help="Number of f_hat->V->f_hat refinement rounds "
                             "(0 = warm-start baseline, no refinement)")
    parser.add_argument("--refine_rho", type=float, default=1.0,
                        help="Weight on the Omega_rob volume penalty")
    parser.add_argument("--refine_objective", type=str, default="volume",
                        choices=["volume", "boundary"],
                        help="Refinement objective: 'volume' (shrink Vol(Omega_rob)) "
                             "or 'boundary' (Direction 2: boundary-weighted gradV-"
                             "projected accuracy)")
    parser.add_argument("--ic_radius", type=float, default=0.0,
                        help="If >0, sample data-collection initial conditions from "
                             "the ball ||x||<=ic_radius (wider than the ||x||<=0.5 "
                             "safe ball) so the data reaches toward the ||x||=1.5 "
                             "cliff. 0 = original near-goal collection.")
    parser.add_argument("--accelerator", type=str, default="cpu",
                        help="cpu, gpu, or auto")
    parser.add_argument("--num_rollouts", type=int, default=100)
    parser.add_argument("--t_sim", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="data/sweep_results.csv")
    parser.add_argument("--robust_delta", type=float, default=0.0,
                        help="If >0, train V with the robust loss (inflate descent "
                             "condition by robust_delta * ||grad V||), testing the "
                             "d_rob ablation.")
    args = parser.parse_args()

    fieldnames = [
        "num_trajectories", "trajectory_length", "N", "obs_noise_std", "seed",
        "max_epochs", "poly_degree", "delta_sup", "delta_mean", "delta_median", "safety_rate",
        "exit_rate", "goal_rate", "limsup_V", "limsup_V_survivors",
        "n_survivors", "num_rollouts", "mean_exit_time", "B", "lambda1",
        "bad_set_fraction", "C_eps_term", "predicted_limsup",
        "safety_aware", "refine_rounds", "refine_vol_before", "refine_vol_after",
        "refine_data_ratio", "refine_cheated", "refine_objective",
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Append-friendly: write header only if the file is new/empty.
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    f = open(args.output, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        f.flush()

    configs = [
        (nt, ns, seed)
        for nt in args.num_trajectories
        for ns in args.obs_noise_std
        for seed in args.seeds
    ]
    print(f"Running {len(configs)} configurations -> {args.output}")

    for i, (nt, ns, seed) in enumerate(configs, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(configs)}] num_traj={nt} noise={ns} seed={seed} ...")
        try:
            row = run_one(
                num_trajectories=nt,
                trajectory_length=args.trajectory_length,
                obs_noise_std=ns,
                seed=seed,
                max_epochs=args.max_epochs,
                accelerator=args.accelerator,
                num_rollouts=args.num_rollouts,
                t_sim=args.t_sim,
                poly_degree=args.poly_degree,
                safety_aware=args.safety_aware,
                refine_rounds=args.refine_rounds,
                refine_rho=args.refine_rho,
                refine_objective=args.refine_objective,
                ic_radius=args.ic_radius,
                robust_delta=args.robust_delta,
            )
        except Exception as exc:  # keep the sweep alive on a bad config
            print(f"    FAILED: {exc!r}")
            continue
        writer.writerow(row)
        f.flush()  # persist incrementally so a crash never loses prior runs
        dt = time.time() - t0
        print(
            f"    N={row['N']} delta={row['delta_sup']:.4f} "
            f"exit_rate={row['exit_rate']:.3f} surv={row['n_survivors']}/{row['num_rollouts']} "
            f"limsupV_surv={row['limsup_V_survivors']:.4f} "
            f"pred={row['predicted_limsup']:.4f} ({dt:.0f}s)"
        )

    f.close()
    print(f"\nSweep complete. Results in {args.output}")


if __name__ == "__main__":
    main()
