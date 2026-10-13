"""
Zero-cost salvage analysis of an existing data/sweep_results.csv.

Nothing here re-runs the sweep; it only re-reads the columns already written and
re-frames them. Two things are fixed relative to plot_tradeoff.py:

  1. Headline metric is the empirical SAFETY RATE (fraction of rollouts that
     never leave the safe set), which carries the real signal -- a sharp cliff.
     The theorem's limsup_V is kept as a secondary panel.

  2. The Theorem 6 prediction is recomputed with the DESIGN rate lambda
     (= CLF_LAMBDA, the lambda the certificate was trained with) instead of the
     empirically fitted lambda_1. The fitted value collapses to its 1e-3 floor
     whenever the undertrained V fails to decay cleanly, and since the bound
     divides by it, that single floor inflates the prediction ~1000x. The
     theorem only requires lambda_1 in (0, lambda], so substituting the design
     lambda is a legitimate, and far tighter, instantiation.

Usage:
    python -m Lyapunov_guided_diffusion.sindy.plot_salvage \
        --input data/sweep_results.csv --output tradeoff_salvage.png
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# The lambda the certificate is trained with (see sweep.py CLF_LAMBDA). The
# theorem guarantees lambda_1 in (0, lambda]; the design value is the tightest
# admissible choice and avoids the collapsed-fit pathology.
CLF_LAMBDA = 1.0


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per noise level over seeds, recomputing the design-lambda bound."""
    # Recompute both bounds row-wise first, then average, so the std reflects
    # seed spread of the bound itself.
    df = df.copy()
    df["pred_original"] = df["C_eps_term"] + df["B"] * df["delta_sup"] / df["lambda1"]
    df["pred_design_sup"] = df["C_eps_term"] + df["B"] * df["delta_sup"] / CLF_LAMBDA
    df["pred_design_mean"] = df["C_eps_term"] + df["B"] * df["delta_mean"] / CLF_LAMBDA

    g = df.groupby("obs_noise_std")
    out = pd.DataFrame({
        "delta_mean": g["delta_mean"].mean(),
        "delta_mean_std": g["delta_mean"].std(),
        "delta_sup": g["delta_sup"].mean(),
        "safety_rate": g["safety_rate"].mean(),
        "safety_rate_std": g["safety_rate"].std(),
        "exit_rate": g["exit_rate"].mean(),
        "goal_rate": g["goal_rate"].mean(),
        "limsup_V": g["limsup_V"].mean(),
        "limsup_V_std": g["limsup_V"].std(),
        "lambda1_fit": g["lambda1"].mean(),
        "pred_original": g["pred_original"].mean(),
        "pred_design_sup": g["pred_design_sup"].mean(),
        "pred_design_mean": g["pred_design_mean"].mean(),
    }).reset_index()
    return out.sort_values("obs_noise_std").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sweep_results.csv")
    parser.add_argument("--output", type=str, default="tradeoff_salvage.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    s = summarize(df)

    # ---- Console summary (the part that decides whether a re-run is needed) ----
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    cols = ["obs_noise_std", "delta_mean", "delta_sup", "safety_rate",
            "goal_rate", "limsup_V", "lambda1_fit",
            "pred_original", "pred_design_sup", "pred_design_mean"]
    print("\n=== Per-noise summary (mean over seeds) ===")
    print(s[cols].to_string(index=False))

    # Locate the cliff: last fully-safe noise vs first fully-unsafe noise.
    safe = s[s["safety_rate"] >= 0.999]
    unsafe = s[s["safety_rate"] <= 0.001]
    if len(safe) and len(unsafe):
        last_safe = safe.iloc[-1]
        first_unsafe = unsafe.iloc[0]
        print("\n=== Safety cliff ===")
        print(f"  last 100%-safe : noise={last_safe['obs_noise_std']:.3f}, "
              f"delta_mean={last_safe['delta_mean']:.2f}")
        print(f"  first 0%-safe  : noise={first_unsafe['obs_noise_std']:.3f}, "
              f"delta_mean={first_unsafe['delta_mean']:.2f}")

    # Bound tightness: ratio bound/measured for the regime where V is meaningful.
    print("\n=== Bound tightness (lower is tighter; >1 means the bound holds) ===")
    for _, r in s.iterrows():
        meas = max(r["limsup_V"], 1e-9)
        ro = r["pred_original"] / meas
        rd = r["pred_design_sup"] / meas
        print(f"  noise={r['obs_noise_std']:.3f}: "
              f"original={ro:.3g}x   design-lambda={rd:.3g}x")

    # ---- Figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: the safety cliff vs model error.
    axA.errorbar(s["delta_mean"], s["safety_rate"],
                 xerr=s["delta_mean_std"], yerr=s["safety_rate_std"],
                 fmt="o-", color="tab:red", capsize=3, lw=2,
                 label="safety rate (never left safe set)")
    axA.plot(s["delta_mean"], s["goal_rate"], "s:", color="tab:green",
             alpha=0.8, label="goal-reached rate")
    axA.set_xlabel(r"SINDy model error $\delta_{\mathrm{mean}}=\mathrm{mean}\|f-\hat f\|$")
    axA.set_ylabel("fraction of rollouts")
    axA.set_ylim(-0.02, 1.05)
    axA.set_title("Empirical safety cliff")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="center right", fontsize=9)

    # Panel B: measured limsup_V vs the (corrected) certificate bound.
    axB.plot(s["delta_mean"], np.maximum(s["limsup_V"], 1e-6), "o-",
             color="tab:blue", lw=2, label=r"measured $\limsup_t V$")
    axB.plot(s["delta_mean"], s["pred_design_sup"], "--", color="tab:orange",
             lw=2, label=r"bound, design $\lambda$ ($\delta_{\sup}$)")
    axB.plot(s["delta_mean"], s["pred_original"], ":", color="gray",
             lw=1.5, label=r"bound, fitted $\lambda_1$ (vacuous)")
    axB.set_yscale("log")
    axB.set_xlabel(r"SINDy model error $\delta_{\mathrm{mean}}$")
    axB.set_ylabel(r"$\limsup_t V(x_t)$  (log scale)")
    axB.set_title("Theorem 6 bound: design $\\lambda$ vs collapsed $\\lambda_1$")
    axB.grid(True, which="both", alpha=0.3)
    axB.legend(loc="center right", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
