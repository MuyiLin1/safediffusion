"""
One configuration of the sharpened-theory experiments (pendulum or cart-pole):
identify f_hat, build a certificate, compute every bound of theory_nd.analyze, and
roll out the true closed loop under three policies. One CSV row per policy.

  python -m Lyapunov_guided_diffusion.sindy.exp_nd --system cartpole --cert quad \
      --sigma 0.1 --num_trajectories 200 --seed 0 --output /tmp/x.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from Lyapunov_guided_diffusion.sindy import theory_nd as T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=list(T.SPECS), required=True)
    p.add_argument("--cert", choices=["quad", "net", "res"], default="quad")
    p.add_argument("--rho", type=float, default=None, help="ridge parameter (default: per system)")
    p.add_argument("--r_scale", type=float, default=2.0, help="envelope inflation in certificate training")
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--num_trajectories", type=int, default=50)
    p.add_argument("--trajectory_length", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--h", type=float, default=0.05)
    p.add_argument("--policies", nargs="+", default=["lqr", "lqr_filter", "lqr_dfilter"])
    p.add_argument("--num_rollouts", type=int, default=30)
    p.add_argument("--t_sim", type=float, default=10.0)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    t0 = time.time()
    spec = T.SPECS[a.system]
    X, U, Y = T.collect(spec, a.num_trajectories, a.trajectory_length, a.sigma, a.seed)
    model, env = T.fit(spec, X, U, Y, max(a.sigma, 1e-4), 15.0, rho=a.rho)
    fit_box = spec.safe_box is not None
    if a.cert == "quad":
        cert = T.quad_cert_from_model(spec, model)
    elif a.cert == "res":
        q = T.quad_cert_from_model(spec, model)
        cq = T.analyze(spec, q, model, env, lam=a.lam, sampled=False, certify=False)["c"]
        cert = T.train_netcert(spec, model, env, a.lam, r_scale=1.5, init=T.ResCert(spec, q.P / cq),
                               size_scale=0.5, size_frac=0.9, w_size=0.1, steps=6000, lr=1e-3,
                               margin=0.1, seed=a.seed, log_every=2000)
    else:
        cert = T.train_netcert(spec, model, env, a.lam, r_scale=a.r_scale, seed=a.seed, log_every=1000)
    base = {"system": a.system, "cert": a.cert, "r_scale": a.r_scale if a.cert == "net" else 0.0,
            "sigma": a.sigma, "num_trajectories": a.num_trajectories, "N": int(X.shape[0]),
            "seed": a.seed, "lam": a.lam, "h": a.h, "kappa": env.kappa, "beta_N": env.beta_N,
            "rho": a.rho if a.rho is not None else spec.rho}
    try:
        g = T.analyze(spec, cert, model, env, lam=a.lam, h=a.h, fit_box=fit_box)
    except T.InvalidCertificate as e:
        rows = [dict(base, valid_cert=0, note=str(e))]
        g = None
    if g is not None:
        xK, VK, sc = g.pop("_xK"), g.pop("_VK"), g.pop("_sc")
        c = g["c"]
        base.update({k: v for k, v in g.items()})
        base["valid_cert"] = 1
        base["tau_hat_over_c"] = g["tau_hat"] / c
        base["sampled_certifies"] = int(g["ult_h"] < c and g["m0_h"] > 0)
        gen = torch.Generator().manual_seed(20_000 + a.seed)
        lim = g["m0_h"] if base["sampled_certifies"] else c
        pool = xK[VK <= lim]
        if pool.shape[0] == 0:
            pool = xK
        x0 = pool[torch.randperm(pool.shape[0], generator=gen)[:a.num_rollouts]]
        rows = []
        for kind in a.policies:
            tp = time.time()
            pi = T.make_policy(kind, spec, cert, model, env, a.lam, sc)
            res = T.rollout(spec, cert, pi, x0, c, t_sim=a.t_sim, h=a.h)
            row = dict(base, policy=kind)
            row["safe_rate"] = float(res["safe"].double().mean())
            row["unsafe_rate"] = float(res["unsafe"].double().mean())
            row["max_V_over_c"] = float(res["Vmax"].max() / c)
            row["tail_max_V"] = float(res["tail"].max())
            row["tail_le_tau_hat"] = int(float(res["tail"].max()) <= g["tau_hat"] + 1e-9)
            if base["sampled_certifies"]:
                k = torch.arange(res["Vk"].shape[1], dtype=torch.float64)
                bound = torch.maximum(torch.exp(-a.lam * a.h * k)[None, :] * res["V0"][:, None],
                                      torch.tensor(max(g["tau_hat_h"], g["J_h"]), dtype=torch.float64))
                row["thm_sampled_safe_ok"] = int(bool((res["Vmax"] < c).all()))
                row["thm_sampled_decay_ok"] = int(bool((res["Vk"] <= bound + 1e-10).all()))
                row["thm_sampled_decay_excess"] = float((res["Vk"] - bound).clamp(min=0).max())
            row["seconds"] = time.time() - tp
            rows.append(row)
            print(f"  {kind:12s} safe={row['safe_rate']:.2f} maxV/c={row['max_V_over_c']:.3f} "
                  f"tail={row['tail_max_V']:.3g} tau_hat={g['tau_hat']:.3g} ult_h={g['ult_h']:.3g} "
                  f"decay_ok={row.get('thm_sampled_decay_ok', '-')} "
                  f"c={c:.3g} ({row['seconds']:.0f}s)", flush=True)
    print(f"total {time.time() - t0:.0f}s")
    fields = sorted({k for r in rows for k in r})
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
