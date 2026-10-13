"""Plot the delta-scope experiment: why the box-restricted model error mis-ranks
safety across data regimes, and how the reachable-set error corrects it at the
safety cliff.

Consumes the two CSVs produced by ``sweep_delta_scope`` (interior/"narrow" and
boundary-aware/"wide" arms) and renders a three-panel figure:

  (A) model error over the certified box (delta_box) vs. noise -- the boundary
      arm carries a *larger* box error at every noise level, yet is safer;
  (B) model error along the path the closed loop actually takes (delta_reach)
      vs. noise -- at the interior cliff the unsafe arm's error jumps above the
      safe arm's, the ordering the box misses;
  (C) the mechanism: peak excursion max||x|| vs. noise, against the unsafe
      boundary ||x||=1.5 -- failures are overshoots into the extrapolation
      region the box never sees.

Safety is encoded by marker fill (filled = safe, open = unsafe); the data arm is
encoded by colour and line style. Bars are +/- one standard deviation across seeds.

This is a read-only analysis/plotting script: it does not touch the sweep code or
the identified models, only the CSVs they emitted.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Repo root relative to this file: .../Lyapunov_guided_diffusion/sindy/ -> repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_DEFAULT_DIR = os.path.join(_REPO_ROOT, "sweep_results", "wide_experiment")

UNSAFE_NORM = 1.5  # ||x|| >= 1.5 is unsafe (matches evaluate.py)
SAFE_THRESHOLD = 0.5  # safety_rate >= this counts as "safe" for the marker fill

NARROW = {"color": "#1f77b4", "ls": "-", "marker": "o", "label": "interior (narrow)"}
WIDE = {"color": "#d62728", "ls": "--", "marker": "s", "label": "boundary-aware (wide)"}


def _read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def _aggregate(rows):
    """Group rows by noise level; return sorted sigmas and per-sigma mean/std dicts."""
    by_sigma = defaultdict(list)
    for r in rows:
        by_sigma[_f(r, "obs_noise_std")].append(r)

    sigmas = sorted(by_sigma)
    out = {}
    for key in (
        "safety_rate",
        "delta_sup_box_1.0",
        "delta_sup_reach",
        "reach_max_norm",
    ):
        means, stds = [], []
        for s in sigmas:
            vals = [_f(r, key) for r in by_sigma[s]]
            vals = [v for v in vals if v == v]  # drop NaNs
            means.append(statistics.mean(vals) if vals else float("nan"))
            stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        out[key] = (means, stds)
    return sigmas, out


def _plot_arm(ax, sigmas, agg, key, style):
    """Line + errorbars for one arm/metric, with marker fill encoding safety."""
    means, stds = agg[key]
    safe_means, _ = agg["safety_rate"]

    ax.errorbar(
        sigmas,
        means,
        yerr=stds,
        color=style["color"],
        ls=style["ls"],
        lw=1.8,
        marker="",
        capsize=3,
        zorder=2,
    )
    # Overlay markers: filled if safe, hollow (white face) if unsafe.
    for x, y, safe in zip(sigmas, means, safe_means):
        filled = safe >= SAFE_THRESHOLD
        ax.plot(
            x,
            y,
            marker=style["marker"],
            markersize=8,
            markeredgecolor=style["color"],
            markerfacecolor=style["color"] if filled else "white",
            markeredgewidth=1.6,
            ls="none",
            zorder=3,
        )


def make_figure(narrow_csv, wide_csv, out_path):
    n_sigmas, n_agg = _aggregate(_read_rows(narrow_csv))
    w_sigmas, w_agg = _aggregate(_read_rows(wide_csv))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax_box, ax_reach, ax_over = axes

    # (A) box-restricted error
    _plot_arm(ax_box, n_sigmas, n_agg, "delta_sup_box_1.0", NARROW)
    _plot_arm(ax_box, w_sigmas, w_agg, "delta_sup_box_1.0", WIDE)
    ax_box.set_title(r"(A) Error over the certified box $\delta_{\mathrm{box}}$")
    ax_box.set_xlabel(r"measurement noise $\sigma$")
    ax_box.set_ylabel(r"$\delta$ over $\|x\|\leq 1$")

    # (B) reachable-set error
    _plot_arm(ax_reach, n_sigmas, n_agg, "delta_sup_reach", NARROW)
    _plot_arm(ax_reach, w_sigmas, w_agg, "delta_sup_reach", WIDE)
    ax_reach.set_title(r"(B) Error along the path $\delta_{\mathrm{reach}}$")
    ax_reach.set_xlabel(r"measurement noise $\sigma$")
    ax_reach.set_ylabel(r"$\delta$ over states reached")

    # (C) overshoot mechanism
    _plot_arm(ax_over, n_sigmas, n_agg, "reach_max_norm", NARROW)
    _plot_arm(ax_over, w_sigmas, w_agg, "reach_max_norm", WIDE)
    ax_over.axhline(UNSAFE_NORM, color="red", ls="--", lw=1.4)
    ax_over.text(
        max(n_sigmas[-1], w_sigmas[-1]),
        UNSAFE_NORM,
        r" unsafe $\|x\|=1.5$",
        va="bottom",
        ha="right",
        color="red",
        fontsize=9,
    )
    ax_over.set_title(r"(C) Peak excursion $\max_t\|x_t\|$")
    ax_over.set_xlabel(r"measurement noise $\sigma$")
    ax_over.set_ylabel(r"$\max_t\|x_t\|$ (true loop)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    # Shared legend: arms + the fill convention.
    handles = [
        Line2D([0], [0], color=NARROW["color"], ls=NARROW["ls"], marker=NARROW["marker"],
               markerfacecolor=NARROW["color"], markersize=8, lw=1.8, label=NARROW["label"]),
        Line2D([0], [0], color=WIDE["color"], ls=WIDE["ls"], marker=WIDE["marker"],
               markerfacecolor=WIDE["color"], markersize=8, lw=1.8, label=WIDE["label"]),
        Line2D([0], [0], color="0.3", marker="o", markerfacecolor="0.3", markersize=8,
               ls="none", label="safe (filled)"),
        Line2D([0], [0], color="0.3", marker="o", markerfacecolor="white", markersize=8,
               markeredgecolor="0.3", ls="none", label="unsafe (open)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--narrow", default=os.path.join(_DEFAULT_DIR, "delta_scope_narrow.csv"))
    ap.add_argument("--wide", default=os.path.join(_DEFAULT_DIR, "delta_scope_wide.csv"))
    ap.add_argument("--output", default=os.path.join(_REPO_ROOT, "delta_scope.png"))
    args = ap.parse_args()
    make_figure(args.narrow, args.wide, args.output)


if __name__ == "__main__":
    main()
