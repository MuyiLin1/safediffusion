"""
Corrected noise-sweep trade-off figure.

Knob = measurement noise (obs_noise_std). For each noise level we aggregate
over seeds and show three panels:

  A. model error delta vs noise          (delta is now a clean, monotone
                                           function of noise -- the +/-1000
                                           control-probe artifact is gone)
  B. safety / goal rate vs delta          (the downstream trade-off: a safety
                                           cliff, with goal-rate degrading
                                           gracefully *before* it)
  C. steady-state lim sup V vs delta      (empirical Theorem-6 trend: the
                                           realized certificate value of
                                           survivors grows monotonically in
                                           delta)

Usage:
    python -m Lyapunov_guided_diffusion.sindy.plot_corrected_tradeoff \
        --input data/corrected_sweep.csv --output corrected_tradeoff.png
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None, "nan") else np.nan)
                         for k, v in r.items()})
    return rows


def agg_by_noise(rows, field):
    """Group a field by obs_noise_std; return sorted noise, mean, std arrays."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["obs_noise_std"]].append(r[field])
    noise = np.array(sorted(groups))
    mean = np.array([np.nanmean(groups[n]) for n in noise])
    std = np.array([np.nanstd(groups[n]) for n in noise])
    return noise, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/corrected_sweep.csv")
    parser.add_argument("--output", type=str, default="corrected_tradeoff.png")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    noise, dsup, dsup_e = agg_by_noise(rows, "delta_sup")
    _, dmean, dmean_e = agg_by_noise(rows, "delta_mean")
    _, safe, safe_e = agg_by_noise(rows, "safety_rate")
    _, goal, goal_e = agg_by_noise(rows, "goal_rate")
    _, vsurv, vsurv_e = agg_by_noise(rows, "limsup_V_survivors")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.8))

    # ---- Panel A: delta vs noise ----
    axA.errorbar(noise, dsup, yerr=dsup_e, fmt="o-", color="tab:purple",
                 capsize=3, label=r"$\delta_{\sup}=\sup\|f-\hat f\|$")
    axA.errorbar(noise, dmean, yerr=dmean_e, fmt="s--", color="tab:orange",
                 capsize=3, label=r"$\delta_{\mathrm{mean}}$")
    axA.set_xlabel("measurement noise std")
    axA.set_ylabel(r"SINDy model error $\delta$")
    axA.set_title(r"(A) $\delta$ grows smoothly with noise")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="upper left", fontsize=9)

    # ---- Panel B: safety + goal vs delta_sup (the theorem's quantity) ----
    order = np.argsort(dsup)
    ds = dsup[order]
    axB.errorbar(ds, safe[order], xerr=dsup_e[order], yerr=safe_e[order],
                 fmt="o-", color="tab:red", capsize=3, label="safety rate")
    axB.errorbar(ds, goal[order], xerr=dsup_e[order], yerr=goal_e[order],
                 fmt="s--", color="tab:green", capsize=3, label="goal-reached rate")
    axB.set_xlabel(r"model error $\delta=\sup\|f-\hat f\|$")
    axB.set_ylabel("rate")
    axB.set_ylim(-0.05, 1.08)
    axB.set_title(r"(B) Safety cliff; goal-rate fades earlier")
    axB.grid(True, alpha=0.3)
    axB.legend(loc="lower left", fontsize=9)

    # ---- Panel C: lim sup V (survivors) vs delta_sup ----
    axC.errorbar(ds, vsurv[order], xerr=dsup_e[order], yerr=vsurv_e[order],
                 fmt="o-", color="tab:blue", capsize=3)
    axC.set_yscale("log")
    axC.set_xlabel(r"model error $\delta=\sup\|f-\hat f\|$")
    axC.set_ylabel(r"steady-state $\limsup_t V$ (survivors)")
    axC.set_title(r"(C) Certificate value rises with $\delta$ (Thm 6 trend)")
    axC.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved {args.output}")

    # Console summary (per-noise, seed-averaged).
    print("\n noise   delta_sup  delta_mean  safety  goal   limsupV_surv")
    for i, n in enumerate(noise):
        print(f"  {n:<6.3g} {dsup[i]:8.2f}   {dmean[i]:8.2f}   "
              f"{safe[i]:5.2f}  {goal[i]:5.2f}  {vsurv[i]:.4f}")


if __name__ == "__main__":
    main()
