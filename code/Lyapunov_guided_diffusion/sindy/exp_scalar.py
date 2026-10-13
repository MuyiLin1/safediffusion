"""
Stage 0: one-dimensional checks of Proposition 3 (tightness) and Proposition 7
(minimax lower bound and matching upper bound). No neural network, no training.

Outputs
  <out_dir>/tightness.csv : for each delta, tau_hat computed on a grid vs the closed
                            forms of Prop. 3, and the simulated limit of V in case (b)
  <out_dir>/lower_bound.csv : for each N, Monte Carlo limsup V of the certainty-
                            equivalent policy vs the lower bound sigma^2/(16(a+k)^2 N)
                            and the upper bound (2 r_N/(2a+2k-lambda))^2 of Prop. 7
"""
import argparse
import csv
import math
import os

import numpy as np


def tightness(out_dir, a=1.0, lam=0.8):
    xs = np.linspace(-1.0, 1.0, 400_001)
    rows = []
    for d in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
        if d > (2 * a - lam) / 4:
            continue
        # (a) f = -a x, f_hat = -a x - d, r = d
        h_a = 2 * xs * (-a * xs - d) + lam * xs ** 2 + 2 * np.abs(xs) * d
        tau_a = float((xs[h_a > 0] ** 2).max()) if (h_a > 0).any() else 0.0
        pred_a = (4 * d / (2 * a - lam)) ** 2
        # (b) f = -a x + d, f_hat = -a x, r = d
        h_b = -(2 * a - lam) * xs ** 2 + 2 * d * np.abs(xs)
        tau_b = float((xs[h_b > 0] ** 2).max()) if (h_b > 0).any() else 0.0
        x = 0.9
        for _ in range(int(40 / 1e-3)):
            x += 1e-3 * (-a * x + d)
        rows.append({"delta": d, "tau_hat_a_grid": tau_a, "tau_delta_a_formula": pred_a,
                     "tau_hat_b_grid": tau_b, "tau_hat_b_formula": (2 * d / (2 * a - lam)) ** 2,
                     "limV_b_sim": x * x, "limV_b_formula": (d / a) ** 2,
                     "ratio_limV_over_tau_b": x * x / tau_b})
    _write(os.path.join(out_dir, "tightness.csv"), rows)
    for r in rows:
        print(f"delta={r['delta']:.3f} (a) grid {r['tau_hat_a_grid']:.3e} formula "
              f"{r['tau_delta_a_formula']:.3e} | (b) limV/tau = {r['ratio_limV_over_tau_b']:.3f}")


def lower_bound(out_dir, a=1.0, k=1.0, lam=0.5, sigma=1.0, theta=0.3, rho=1e-2,
                S=5.0, alpha=0.1, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for N in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        # Ridge on the constant feature after subtracting the known part f0 = -a x + u.
        eps = rng.standard_normal((reps, N)) * sigma
        theta_hat = (theta * N + eps.sum(axis=1)) / (N + rho)
        r_N = (sigma * math.sqrt(2 * math.log(1 / alpha) + math.log(1 + N / rho))
               + math.sqrt(rho) * S) / math.sqrt(N + rho)
        # Closed loop x' = -(a+k) x + (theta - theta_hat): limit x_e = (theta-theta_hat)/(a+k).
        limV = ((theta - theta_hat) / (a + k)) ** 2
        # Simulate a few to confirm the closed form.
        x = np.full(min(reps, 50), 0.4)
        e = theta - theta_hat[:x.size]
        for _ in range(int(20 / 1e-3)):
            x += 1e-3 * (-(a + k) * x + e)
        upper = (2 * r_N / (2 * a + 2 * k - lam)) ** 2
        # Two-point minimax value at Delta = sigma / sqrt(N): theta in {0, Delta}.
        Delta = sigma / math.sqrt(N)
        two_pt = []
        for th in (0.0, Delta):
            th_hat = (th * N + eps.sum(axis=1)) / (N + rho)
            two_pt.append(float(np.mean(((th - th_hat) / (a + k)) ** 2)))
        rows.append({
            "N": N, "mean_limV": float(limV.mean()),
            "q90_limV": float(np.quantile(limV, 1 - alpha)),
            "lower_bound": sigma ** 2 / (16 * (a + k) ** 2 * N),
            "upper_bound": upper,
            "frac_below_upper": float((limV <= upper).mean()),
            "envelope_coverage": float((np.abs(theta - theta_hat) <= r_N).mean()),
            "two_point_max_mean": max(two_pt),
            "sim_vs_formula_maxerr": float(np.abs(x ** 2 - limV[:x.size]).max()),
        })
    _write(os.path.join(out_dir, "lower_bound.csv"), rows)
    for r in rows:
        print(f"N={r['N']:5d} lower={r['lower_bound']:.2e} mean={r['mean_limV']:.2e} "
              f"upper={r['upper_bound']:.2e} below_upper={r['frac_below_upper']:.3f} "
              f"cover={r['envelope_coverage']:.3f}")


def _write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/theory/stage0_scalar")
    args = p.parse_args()
    tightness(args.out_dir)
    lower_bound(args.out_dir)


if __name__ == "__main__":
    main()
