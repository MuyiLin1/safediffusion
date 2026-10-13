"""
Merge per-job CSVs and turn the theory-validation results into figures and a
pass/fail summary.

  python -m Lyapunov_guided_diffusion.sindy.plot_theory merge <parts_dir> <out_csv>
  python -m Lyapunov_guided_diffusion.sindy.plot_theory report --root data/theory --fig_dir figures/theory
"""
import argparse
import csv
import glob
import math
import os

import numpy as np

FILTER_POLICIES = ("qp_filter", "diffusion_filter")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def merge(parts_dir, out_csv):
    rows, fields = [], []
    for path in sorted(glob.glob(os.path.join(parts_dir, "*.csv"))):
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
                for k in r:
                    if k not in fields:
                        fields.append(k)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(rows)} rows from {parts_dir} -> {out_csv}")


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "nan")))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.array(out)


def where(rows, **kw):
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]


def slope(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def group_median(rows, key_x, key_y):
    xs = sorted({float(r[key_x]) for r in rows})
    ys = []
    for v in xs:
        sel = col([r for r in rows if float(r[key_x]) == v], key_y)
        sel = sel[np.isfinite(sel)]
        ys.append(float(np.median(sel)) if sel.size else float("nan"))
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(root, fig_dir, alpha=0.1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    S = {name: load(os.path.join(root, f"{name}.csv")) for name in
         ["stage2_noise_fixed", "stage3_samples_fixed", "stage4_noise_retrain",
          "stage5_coverage_retrain", "stage6_misspec_fixed", "stage7_eiv_fixed",
          "stage10_cert_arms", "stage11_diffusion", "stage12_diffusion_matched"]}
    n_invalid = {k: sum(r.get("valid_cert") == "0" for r in v) for k, v in S.items()}
    S = {k: [r for r in v if r.get("valid_cert") != "0"] for k, v in S.items()}
    tight = load(os.path.join(root, "stage0_scalar", "tightness.csv"))
    lower = load(os.path.join(root, "stage0_scalar", "lower_bound.csv"))
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- Stage 0 ---------------------------------------------------------
    if tight and lower:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        d = col(tight, "delta")
        ax[0].loglog(d, col(tight, "tau_hat_a_grid"), "o", label=r"(a) $\hat\tau$ on grid")
        ax[0].loglog(d, col(tight, "tau_delta_a_formula"), "-", label=r"(a) $\tau_\delta$ (Thm 2)")
        ax[0].loglog(d, col(tight, "limV_b_sim"), "s", label=r"(b) simulated $\lim V$")
        ax[0].loglog(d, col(tight, "tau_hat_b_grid") / 4, "--", label=r"(b) $\hat\tau/4$")
        ax[0].set_xlabel(r"$\delta$")
        ax[0].set_title("Prop. 3: tightness")
        ax[0].legend(fontsize=8)
        N = col(lower, "N")
        ax[1].loglog(N, col(lower, "lower_bound"), "k--", label="lower bound (Prop. 7)")
        ax[1].loglog(N, col(lower, "mean_limV"), "o-", label=r"mean $\limsup V$")
        ax[1].loglog(N, col(lower, "q90_limV"), "^-", label=r"90% quantile")
        ax[1].loglog(N, col(lower, "upper_bound"), "r-", label="upper bound (Prop. 7)")
        ax[1].set_xlabel("N")
        ax[1].set_title("Prop. 7: 1/N rate")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_scalar.png"), dpi=150)
        plt.close(fig)
        err_a = np.max(np.abs(col(tight, "tau_hat_a_grid") / col(tight, "tau_delta_a_formula") - 1))
        say("== Stage 0 (1-D) ==")
        say(f"  Prop 3(a) grid tau_hat vs formula: max relative error {err_a:.2e} (expect ~0)")
        say(f"  Prop 3(b) min limV/tau_hat: {np.min(col(tight, 'ratio_limV_over_tau_b')):.3f} (expect >= 0.25)")
        say(f"  Prop 7 slope of mean limsup V vs N: {slope(N, col(lower, 'mean_limV')):.2f} (expect -1)")
        say(f"  Prop 7 mean >= lower bound at every N: {bool(np.all(col(lower, 'mean_limV') >= col(lower, 'lower_bound')))}")
        say(f"  Prop 7 min fraction below upper bound: {np.min(col(lower, 'frac_below_upper')):.3f} (expect >= {1 - alpha})")

    say("")
    say("== Configs skipped because V had no safe sublevel set around the goal ==")
    for name, k in n_invalid.items():
        if k:
            say(f"  {name:26s} {k}")

    # ---- Envelope validity (Prop. 5) -------------------------------------
    say("\n== Envelope validity (Prop. 5): fraction of configs where |f - f_hat| <= r_N on the whole grid ==")
    for name, rows in S.items():
        rows = _one_per_config(rows)
        if rows:
            v = col(rows, "env_valid")
            say(f"  {name:26s} {np.nanmean(v):.3f} over {len(rows)} configs "
                f"(theory: >= {1 - alpha} when noise_on=xdot; stage7 is outside the theory)")

    # ---- Theorem 2 -------------------------------------------------------
    say("\n== Theorem 2: tau_hat <= tau_delta whenever lambda_bar > lambda and the envelope is valid ==")
    for name, rows in S.items():
        rows = _one_per_config(rows)
        app = [r for r in rows if r.get("thm2_applicable") == "1" and r.get("env_valid") == "1"]
        if app:
            say(f"  {name:26s} holds in {np.mean(col(app, 'thm2_holds')):.3f} of {len(app)} applicable configs "
                f"(theory: 1.000)")

    # ---- Theorem 1 -------------------------------------------------------
    say("\n== Theorem 1 (ideal form): with a filter, tau_hat < c should imply safety ==")
    all_rows = [r for rows in S.values() for r in rows
                if r.get("safe_V_rate") not in (None, "") and r.get("env_valid") == "1"]
    broken = [r for rows in S.values() for r in rows
              if r.get("safe_V_rate") not in (None, "") and r.get("env_valid") == "0"]
    if broken:
        bp = [r for r in broken if r["pred_safe_ideal"] == "1"]
        say(f"  (excluded: {len(broken)} rows whose envelope missed the true error; of the "
            f"{len(bp)} 'predicted safe' among them, fully safe {np.mean(col(bp, 'safe_V_rate') == 1.0) if bp else float('nan'):.3f})")
    for pol in ["qp", "qp_filter", "diffusion", "diffusion_filter"]:
        sel = [r for r in all_rows if r["policy"] == pol]
        if not sel:
            continue
        ps = [r for r in sel if r["pred_safe_ideal"] == "1"]
        safeV = col(ps, "safe_V_rate")
        say(f"  {pol:17s} predicted-safe configs: {len(ps):3d}; of these fully safe (V<c always): "
            f"{np.mean(safeV == 1.0) if ps else float('nan'):.3f}; "
            f"min cert_frac {np.nanmin(col(sel, 'cert_frac')):.3f}; "
            f"median excess over max(e^-lt V0, tau_hat): {np.nanmedian(col(sel, 'bound_excess_ideal')):.2e}; "
            f"limsup<=tau_hat in {np.nanmean(col(ps, 'limsup_le_tau_hat')) if ps else float('nan'):.3f}")
    samp = [r for r in all_rows if r.get("pred_safe_sampled") == "1"]
    say(f"  configs where the SAMPLED-DATA form of Thm 1 is non-vacuous: {len(samp)} of {len(all_rows)} "
        f"(expected ~0 at h = 0.05; plan item TH12)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for pol, mk in [("qp", "o"), ("qp_filter", "s"), ("diffusion", "^"), ("diffusion_filter", "D")]:
        sel = [r for r in all_rows if r["policy"] == pol]
        if not sel:
            continue
        t, y = col(sel, "tau_hat"), col(sel, "tail_max_V")
        ax[0].loglog(np.maximum(t, 1e-8), np.maximum(y, 1e-8), mk, alpha=0.5, label=pol)
        ax[1].plot(col(sel, "tau_hat_over_c"), col(sel, "safe_V_rate"), mk, alpha=0.5, label=pol)
    lim = [1e-8, 1e2]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel(r"$\hat\tau$")
    ax[0].set_ylabel(r"tail max $V$")
    ax[0].set_title(r"Thm 1: $\limsup V \leq \hat\tau$ (below the line)")
    ax[1].axvline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xscale("symlog", linthresh=1e-3)
    ax[1].set_xlabel(r"$\hat\tau / c$")
    ax[1].set_ylabel(r"fraction with $V<c$ always")
    ax[1].set_title("Thm 1: left of the line should be safe")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_thm1.png"), dpi=150)
    plt.close(fig)

    # ---- Rates (Thm 2, Cor 6, floor) -------------------------------------
    s2 = _one_per_config(S["stage2_noise_fixed"])
    s3 = _one_per_config(S["stage3_samples_fixed"])
    s6 = _one_per_config(S["stage6_misspec_fixed"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if s3:
        x, y = group_median(s3, "N", "delta_c_env")
        _, yt = group_median(s3, "N", "delta_c_true")
        _, tau = group_median(s3, "N", "tau_hat")
        ax[0].loglog(x, y, "o-", label=r"$\delta_c$ envelope")
        ax[0].loglog(x, yt, "s-", label=r"$\delta_c$ true error")
        ax[1].loglog(x, np.maximum(tau, 1e-10), "o-", label=r"$\hat\tau$")
        say(f"\n== Cor 6 rates (stage 3, fixed V) ==")
        say(f"  slope delta_c(env) vs N: {slope(x, y):.2f} (expect about -0.5, up to log N)")
        say(f"  slope delta_c(true) vs N: {slope(x, yt):.2f}")
        say(f"  slope tau_hat vs N: {slope(x, tau):.2f} (expect about -1 once tau_hat > 0)")
    if s6:
        for lib, mk in [("bias", "^"), ("linear", "v")]:
            sel = [r for r in s6 if r["library"] == lib]
            if sel:
                x, y = group_median(sel, "N", "delta_c_true")
                ax[0].loglog(x, y, mk + "--", label=f"true error, {lib} lib (stage 6)")
                say(f"  stage 6 {lib:6s}: true error at smallest/largest N: {y[0]:.3g} -> {y[-1]:.3g} "
                    f"(linear lib should plateau at the misspecification floor)")
    if s2:
        d, t = col(s2, "delta_c_env"), col(s2, "tau_hat")
        loc = col(s2, "tau_hat_over_c") < 0.25
        ax[2].loglog(d, np.maximum(t, 1e-10), "o", alpha=0.6, label=r"$\hat\tau$")
        ax[2].loglog(d, col(s2, "tau_delta"), "x", alpha=0.6, label=r"$\tau_\delta$ (Thm 2)")
        say(f"\n== Thm 2 rate (stage 2, fixed V) ==")
        say(f"  slope tau_hat vs delta_c(env), configs with tau_hat < c/4: {slope(d[loc], t[loc]):.2f} "
            f"(expect about 2; Thm 2 is an upper bound, so faster decay is allowed)")
    for a, ttl in zip(ax, [r"Prop 5 / Cor 6: error vs $N$", r"Cor 6: $\hat\tau$ vs $N$",
                           r"Thm 2: $\hat\tau$ vs $\delta_c$"]):
        a.set_title(ttl)
        a.legend(fontsize=8)
    ax[0].set_xlabel("N")
    ax[1].set_xlabel("N")
    ax[2].set_xlabel(r"$\delta_c$ (envelope)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_rates.png"), dpi=150)
    plt.close(fig)

    # ---- delta_star vs measured cliff (stage 4) ---------------------------
    s4 = S["stage4_noise_retrain"]
    if s4:
        say("\n== Stage 4 (retrain): measured cliff vs delta_star of Thm 2 ==")
        for pol in sorted({r["policy"] for r in s4}):
            sel = [r for r in s4 if r["policy"] == pol]
            sig, safe = group_median(sel, "sigma", "safe_box_rate")
            _, dce = group_median(sel, "sigma", "delta_c_env")
            _, dst = group_median(sel, "sigma", "delta_star")
            first_bad = next((i for i, v in enumerate(safe) if v < 1.0), None)
            msg = "no failure in grid" if first_bad is None else \
                f"first unsafe sigma={sig[first_bad]:g} with delta_c={dce[first_bad]:.3g}"
            say(f"  {pol:17s} {msg}; median delta_star over grid {np.nanmedian(dst):.3g} "
                f"(Thm 2 is sufficient: safety is guaranteed while delta_c < delta_star)")

    # ---- Coverage (stage 5, Cor 4) ----------------------------------------
    s5 = S["stage5_coverage_retrain"]
    if s5:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        say("\n== Stage 5 (Cor 4): narrow vs wide data ==")
        for icr, mk in [("0.0", "o-"), ("1.4", "s-")]:
            sel = [r for r in _one_per_config(s5) if r["ic_radius"] == icr]
            if not sel:
                continue
            x, rb = group_median(sel, "sigma", "r_band")
            _, tc = group_median(sel, "sigma", "tau_hat_over_c")
            ax[0].semilogy(x, rb, mk, label=f"ic_radius={icr}")
            ax[1].semilogy(x, np.maximum(tc, 1e-6), mk, label=f"ic_radius={icr}")
            say(f"  ic_radius={icr}: median r on top band {np.nanmedian(rb):.3g}, "
                f"median tau_hat/c {np.nanmedian(tc):.3g}")
            for pol in ["qp", "qp_filter"]:
                sp = [r for r in s5 if r["ic_radius"] == icr and r["policy"] == pol]
                if sp:
                    xs, sf = group_median(sp, "sigma", "safe_box_rate")
                    ax[2].plot(xs, sf, mk, label=f"{pol}, ic={icr}")
        ax[0].set_title("envelope on the top band (Cor 4)")
        ax[1].set_title(r"$\hat\tau / c$")
        ax[2].set_title("safety rate (box)")
        for a in ax:
            a.set_xlabel(r"$\sigma$")
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_coverage.png"), dpi=150)
        plt.close(fig)

    # ---- Nominal vs robust certificate training (stage 10) ------------------
    s10 = S["stage10_cert_arms"]
    if s10:
        say("\n== Stage 10: certificate trained on f_hat alone (nominal) vs with the envelope (robust) ==")
        say("  arm             sigma  lam_bar  tau_hat/c  omega_frac  safe(qp)  safe(filter)")
        for arm in ("learned_nominal", "learned_robust"):
            for sg in sorted({float(r["sigma"]) for r in s10}):
                g = [r for r in s10 if r["cert"] == arm and float(r["sigma"]) == sg]
                q = [r for r in g if r["policy"] == "qp"]
                fl = [r for r in g if r["policy"] == "qp_filter"]
                if not q:
                    continue
                say(f"  {arm:15s} {sg:5g}  {np.nanmedian(col(q, 'lambda_bar')):7.3f}  "
                    f"{np.nanmedian(col(q, 'tau_hat_over_c')):9.4f}  {np.nanmedian(col(q, 'omega_frac_Kc')):10.4f}  "
                    f"{np.nanmean(col(q, 'safe_V_rate')):8.3f}  {np.nanmean(col(fl, 'safe_V_rate')) if fl else float('nan'):12.3f}")

    # ---- Diffusion sampler (stage 11) -----------------------------------------
    for stage_name, title in (("stage11_diffusion", "sampler control set +-1000"),
                              ("stage12_diffusion_matched", "sampler control set matched to U")):
        s11 = S[stage_name]
        if not s11:
            continue
        say(f"\n== {stage_name}: diffusion sampler guided by the learned certificate ({title}) ==")
        for pol in ("diffusion", "diffusion_filter"):
            for sg in sorted({float(r["sigma"]) for r in s11}):
                g = [r for r in s11 if r["policy"] == pol and float(r["sigma"]) == sg]
                if g:
                    say(f"  {pol:17s} sigma={sg:g}: tau_hat/c {np.nanmedian(col(g, 'tau_hat_over_c')):.4f}, "
                        f"safe {np.nanmean(col(g, 'safe_V_rate')):.3f}, cert_frac {np.nanmean(col(g, 'cert_frac')):.3f}, "
                        f"tail V/c {np.nanmedian(col(g, 'tail_max_V') / col(g, 'c')):.4f}")

    with open(os.path.join(fig_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nfigures and summary.txt written to {fig_dir}")


def _one_per_config(rows):
    """Geometry columns repeat once per policy; keep one row per configuration."""
    seen, out = set(), []
    for r in rows:
        key = (r.get("library"), r.get("noise_on"), r.get("sigma"), r.get("sigma_x"),
               r.get("num_trajectories"), r.get("ic_radius"), r.get("seed"), r.get("mode"),
               r.get("cert"))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("parts_dir")
    m.add_argument("out_csv")
    r = sub.add_parser("report")
    r.add_argument("--root", default="data/theory")
    r.add_argument("--fig_dir", default="figures/theory")
    args = p.parse_args()
    if args.cmd == "merge":
        merge(args.parts_dir, args.out_csv)
    else:
        report(args.root, args.fig_dir)


if __name__ == "__main__":
    main()
