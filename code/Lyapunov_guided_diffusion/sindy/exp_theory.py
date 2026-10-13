"""
One configuration of the theory-validation experiments (paperv2.tex, Sections 3-4).

Modes
  train_ref : train a reference certificate V on a clean, large dataset and save it.
  fixed     : load the reference V, re-identify f_hat from new data, and test the
              theorems with V held fixed (Theorem 2 is a statement about a fixed V).
  retrain   : full pipeline, V is retrained against the new f_hat.

Each run writes one CSV row per policy in --policies to --output.

Example
  python -m Lyapunov_guided_diffusion.sindy.exp_theory --mode fixed \
      --ref_v data/theory/ref_v/ref_seed0.pt --sigma 0.5 --num_trajectories 50 \
      --seed 0 --policies qp qp_filter --output /tmp/one.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from neural_clbf.systems.sindy_estimated_system import SINDyEstimatedSystem

from Lyapunov_guided_diffusion.sindy import robust_certificate as RC
from Lyapunov_guided_diffusion.sindy.collect_data import collect_safe_transitions
from Lyapunov_guided_diffusion.sindy.theory_tools import (
    RobustCertificate,
    CertView,
    InvalidCertificate,
    analyze_geometry,
    fit_ridge,
    make_policy,
    make_true_system,
    make_untrained_controller,
    oracle_misspecification,
    rollout,
    train_certificate,
)

SIM_DT = 0.01
CONTROLLER_PERIOD = 0.05


def collect(true_system, args, seed):
    """Exact derivative targets along closed-loop trajectories, then additive
    Gaussian noise: on the targets only (noise_on=xdot, the setting of Prop. 5) or
    also on the recorded states (noise_on=both, errors-in-variables)."""
    X, U, Xdot = collect_safe_transitions(
        true_system,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        exploration_std=0.5,
        obs_noise_std=0.0,
        seed=seed,
        ic_radius=args.ic_radius,
    )
    rng = np.random.default_rng(10_000 + seed)
    Y = Xdot + args.sigma * rng.standard_normal(Xdot.shape)
    Xr = X.copy()
    if args.noise_on == "both":
        Xr = X + args.sigma_x * rng.standard_normal(X.shape)
    return Xr, U, Y, X


def identify(X, U, Y, args):
    m_bar, S = 0.0, args.S
    Xi_star_norm = float("nan")
    if args.library == "linear":
        Xi_star, m_bar = oracle_misspecification(args.library, args.u_max)
        Xi_star_norm = float(np.linalg.norm(Xi_star, axis=0).max())
        S = max(S, Xi_star_norm)
    model, env = fit_ridge(X, U, Y, args.library, sigma=args.sigma, S=S,
                           alpha=args.alpha, rho=args.rho, m_bar=m_bar)
    return model, env, m_bar, S


def sample_x0(xK, n, seed):
    """Initial states from the grid points of K_c (so V(x0) <= c, the hypothesis of
    Theorem 1), preferring the safe ball ||x|| <= 0.5."""
    g = torch.Generator().manual_seed(20_000 + seed)
    ball = xK[xK.norm(dim=-1) <= 0.5]
    pool, source = (ball, "ball") if ball.shape[0] >= n else (xK, "Kc")
    idx = torch.randperm(pool.shape[0], generator=g)[:n]
    return pool[idx], source


def build_controller(sindy_system, scenarios, args):
    if args.mode in ("retrain", "train_ref"):
        controller = train_certificate(
            sindy_system, scenarios, max_epochs=args.max_epochs,
            accelerator=args.accelerator, barrier=not args.no_barrier,
        )
    else:
        controller = make_untrained_controller(sindy_system, scenarios, CONTROLLER_PERIOD)
        if not args.cert.startswith("learned"):
            state = torch.load(args.ref_v, map_location="cpu")
            controller.V_nn.load_state_dict(state["V_nn"])
    controller = controller.to("cpu")
    controller.eval()
    return controller


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    true_system, scenarios = make_true_system(SIM_DT, CONTROLLER_PERIOD)

    if args.mode == "train_ref":
        args.sigma, args.noise_on, args.library = 0.0, "xdot", "exact"
    X, U, Y, X_clean = collect(true_system, args, args.seed)
    model, env, m_bar, S_used = identify(X, U, Y, args)
    sindy_system = SINDyEstimatedSystem(
        model, reference_system=true_system, dt=SIM_DT,
        controller_dt=CONTROLLER_PERIOD, scenarios=scenarios,
    )
    if args.match_u_limits:
        sindy_system._control_limits = (torch.tensor([args.u_max]), torch.tensor([-args.u_max]))

    lam = args.cert_lambda
    cert_stats = {}
    if args.mode == "train_cert":
        lcert, cert_stats = RC.train_certificate(
            model, env, robust=True, lam=lam, u_max=args.u_max, steps=args.cert_steps, seed=args.seed)
        os.makedirs(os.path.dirname(args.cert_file) or ".", exist_ok=True)
        RC.save_certificate(lcert, args.cert_file, {"sigma": args.sigma, "N": int(X.shape[0]),
                                                    "seed": args.seed, **cert_stats})
        print(f"saved learned certificate -> {args.cert_file} ({time.time() - t0:.0f}s)")
        return

    controller = build_controller(sindy_system, scenarios, args)

    if args.mode == "train_ref":
        os.makedirs(os.path.dirname(args.ref_v) or ".", exist_ok=True)
        torch.save({"V_nn": controller.V_nn.state_dict(), "seed": args.seed,
                    "num_trajectories": args.num_trajectories,
                    "max_epochs": args.max_epochs}, args.ref_v)
        print(f"saved reference V -> {args.ref_v} ({time.time() - t0:.0f}s)")
        return

    cview = CertView(controller, args.cert, args.shift_goal)
    if args.cert == "learned":
        cview = RC.load_certificate(args.cert_file)
    elif args.cert in ("learned_nominal", "learned_robust"):
        cview, cert_stats = RC.train_certificate(
            model, env, robust=args.cert == "learned_robust", lam=lam, u_max=args.u_max,
            steps=args.cert_steps, seed=args.seed)
    if args.cert.startswith("learned"):
        RC.attach(controller, cview)
    cert = RobustCertificate(cview, model, env, lam, args.u_max)
    try:
        geo = analyze_geometry(cview, cert, env, lam, args.u_max, n_grid=args.n_grid)
    except InvalidCertificate as e:
        print(f"  invalid certificate, no theorem applies: {e}")
        row = {"mode": args.mode, "library": args.library, "noise_on": args.noise_on,
               "sigma": args.sigma, "num_trajectories": args.num_trajectories,
               "ic_radius": args.ic_radius, "seed": args.seed, "valid_cert": 0,
               "note": str(e)}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(row))
            w.writeheader()
            w.writerow(row)
        return
    c, tau_hat = geo["c"], geo["tau_hat"]
    xK = geo.pop("_xK")
    h = CONTROLLER_PERIOD
    B = geo["B_loc"]
    tau_h = tau_hat + B ** 2 * h
    e_sampled = B ** 2 * (lam + 2 * B) * h

    x0, x0_source = sample_x0(xK, args.num_rollouts, args.seed)
    base = {
        "mode": args.mode, "library": args.library, "noise_on": args.noise_on,
        "sigma": args.sigma, "sigma_x": args.sigma_x if args.noise_on == "both" else 0.0,
        "num_trajectories": args.num_trajectories, "N": int(X.shape[0]),
        "ic_radius": args.ic_radius, "seed": args.seed, "cert_lambda": lam, "cert": args.cert, "shift_goal": int(args.shift_goal), "match_u_limits": int(args.match_u_limits),
        "u_max": args.u_max, "alpha": args.alpha, "rho": args.rho, "S": S_used,
        "m_bar": m_bar, "beta_N": env.beta_N, "c_N": env.c_N, "kappa": env.kappa,
        "ref_v": os.path.basename(args.ref_v) if args.mode == "fixed" else "",
        "n_x0": int(x0.shape[0]), "x0_source": x0_source,
        "tau_h_sampled": tau_h, "e_sampled": e_sampled,
        "pred_safe_ideal": int(tau_hat < c),
        "pred_safe_sampled": int(max(tau_h, e_sampled / lam) < c),
        "tau_hat_over_c": tau_hat / c if c > 0 else float("nan"),
        "valid_cert": 1,
        **{f"cert_{k}": v for k, v in cert_stats.items()},
    }
    base.update(geo)

    rows = []
    for kind in args.policies:
        tp = time.time()
        pol = make_policy(kind, controller, cert)
        res = rollout(pol, cert, x0, c, tau_hat, lam, args.t_sim, h, SIM_DT) \
            if x0.shape[0] else None
        row = dict(base)
        row["policy"] = kind
        if res is None:
            row["note"] = "no initial state with V(x0) < c"
        else:
            for k in ("safe_box_rate", "safe_V_rate", "diverged_rate", "max_V",
                      "tail_max_V", "limsup_V_mean", "max_norm", "cert_frac",
                      "s_emp", "bound_excess_ideal"):
                row[k] = res[k]
            row["limsup_le_tau_hat"] = int(res["tail_max_V"] <= max(tau_hat, 0.0) + 1e-9)
            row["thm1_sampled_holds"] = int(
                res["max_V"] <= max(res["V0"].max().item(), tau_h, e_sampled / lam) + 1e-9)
        row["seconds"] = time.time() - tp
        rows.append(row)
        print(f"  {kind:17s} safe_box={row.get('safe_box_rate', float('nan')):.2f} "
              f"safe_V={row.get('safe_V_rate', float('nan')):.2f} "
              f"cert_frac={row.get('cert_frac', float('nan')):.3f} "
              f"tailmaxV={row.get('tail_max_V', float('nan')):.4g} tau_hat={tau_hat:.4g} "
              f"c={c:.4g} ({row['seconds']:.0f}s)")
    print(f"  delta_c env={geo['delta_c_env']:.4g} true={geo['delta_c_true']:.4g} "
          f"env_valid={geo['env_valid']} kappa={env.kappa:.3g} total {time.time() - t0:.0f}s")

    fields = sorted({k for r in rows for k in r})
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["train_ref", "train_cert", "fixed", "retrain"], required=True)
    p.add_argument("--ref_v", default="data/theory/ref_v/ref_seed0.pt",
                   help="reference V file (written by train_ref, read by fixed)")
    p.add_argument("--library", choices=["exact", "bias", "rich", "linear"], default="exact")
    p.add_argument("--noise_on", choices=["xdot", "both"], default="xdot")
    p.add_argument("--sigma", type=float, default=0.0, help="derivative-target noise std")
    p.add_argument("--sigma_x", type=float, default=0.0, help="state noise std (noise_on=both)")
    p.add_argument("--num_trajectories", type=int, default=50)
    p.add_argument("--trajectory_length", type=int, default=100)
    p.add_argument("--ic_radius", type=float, default=0.0, help="0 = narrow, e.g. 1.4 = wide")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policies", nargs="+", default=["qp", "qp_filter"],
                   choices=["qp", "qp_filter", "diffusion", "diffusion_filter"])
    p.add_argument("--cert_lambda", type=float, default=0.5, help="lambda in d_rob")
    p.add_argument("--u_max", type=float, default=30.0, help="U = [-u_max, u_max]")
    p.add_argument("--alpha", type=float, default=0.1, help="envelope failure probability")
    p.add_argument("--rho", type=float, default=1e-2, help="ridge parameter")
    p.add_argument("--S", type=float, default=15.0, help="prior bound on column norms of Xi")
    p.add_argument("--n_grid", type=int, default=161)
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--no_barrier", action="store_true",
                   help="train V without the safe/unsafe level terms (as in sweep.py)")
    p.add_argument("--cert", choices=["nn", "quad", "learned", "learned_nominal", "learned_robust"],
                   default="nn",
                   help="certificate checked by the theorems: trained CLBF, fixed quadratic, a saved "
                        "learned certificate (--cert_file), or one trained here on this data")
    p.add_argument("--cert_file", default="", help="learned certificate file (train_cert writes it)")
    p.add_argument("--cert_steps", type=int, default=3000)
    p.add_argument("--match_u_limits", action="store_true",
                   help="give the controller and diffusion sampler the control set U = [-u_max, u_max]")
    p.add_argument("--shift_goal", action="store_true",
                   help="use V - V(goal) so the certificate is 0 at the goal")
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--num_rollouts", type=int, default=30)
    p.add_argument("--t_sim", type=float, default=10.0)
    p.add_argument("--output", default="/tmp/exp_theory.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
