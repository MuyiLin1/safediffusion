# safediffusion: project export

A cleaned-up snapshot of the `safediffusion` research repo, taken on 2026-10-12. It holds
all of the code, the experiment scripts, every result, figure and log, and the trained
models. Paper drafts are left out on purpose.

**If you only read one thing per folder:**
`docs/PAPER_PLAN.md` explains the research goal and its status,
`results/3_theory_validation/theory_learned/summary.txt` gives the latest results,
and `code/Lyapunov_guided_diffusion/sindy/` holds the code behind them.

---

## The project in one paragraph

A controller for a robot (here an inverted pendulum, plus a cart-pole and a two-pole
system) samples control actions with a **diffusion model**. A learned
**Lyapunov/barrier certificate V** guides that sampling, which keeps the robot safe
(V stays below a level c) and moves it to the goal (V decreases). The research question
is: **what happens when the true physics is unknown and has to be learned from noisy
data with SINDy?** The project answers it in three phases, and the folders below are
numbered to match them:

1. **Original method:** diffusion + certificate controller with *known* physics.
2. **Learned physics, measured empirically:** swap in a SINDy model and measure how
   safety falls as the model error δ grows.
3. **Learned physics, with proofs:** prove bounds on how close to the goal the robot
   ends up (of order δ², or 1/N with N data points) and run experiments that check each
   bound. This is the current work.

---

## Folder map

```
export/
├── README.md                ← this file
├── code/                    ← all Python source + requirements.txt
├── scripts/                 ← shell scripts that launched every experiment
├── docs/                    ← plan + plain-language explanations
├── results/                 ← every result CSV and saved model, by phase
│   ├── 1_sindy_pipeline/
│   ├── 2_learned_dynamics_sweeps/
│   └── 3_theory_validation/
├── figures/                 ← every generated image, by phase
├── logs/                    ← console logs + TensorBoard training runs
└── git_info/                ← git history and uncommitted code changes
```

---

## `code/`

`code/Lyapunov_guided_diffusion/` is the Python package. Run commands from `code/`, for
example `python -m Lyapunov_guided_diffusion.sindy.exp_theory --help`. Dependencies are
in `code/requirements.txt` (torch 2.1, pytorch-lightning, pysindy, cvxpy/cvxpylayers).

| Folder | What it contains | Phase |
|---|---|---|
| `controllers/` | **`diffusion_clbf.py`: the main method.** It samples control sequences with a diffusion process guided by the certificate, and trains the certificate. `clf_controller.py` is a QP-based baseline. | 1 |
| `systems/` | Physics models. `inverted_pendulum.py` is the true system. `sindy_estimated_system.py` wraps a SINDy-learned model so it can stand in for the true one. `control_affine_system.py` is the base class. | 1–2 |
| `training/` | Train the controller on the true pendulum (`train_inverted_pendulum.py`) or on SINDy physics (`..._sindy.py`). | 1–2 |
| `datamodules/` | PyTorch Lightning data loading. | 1 |
| `experiments/` | Plotting and rollout helpers (contour plots, rollouts, success rate). | 1 |
| `sindy/` | **All learned-physics work.** See the two tables below. | 2–3 |

**`sindy/`, phase 2 files:**

| File | Does |
|---|---|
| `collect_data.py` → `identify_dynamics.py` → `evaluate.py` | Pipeline: collect data from the true system, fit a SINDy model, then test the controller on the true system. |
| `sweep.py` | Repeats that pipeline over a grid of noise levels, data sizes and seeds. |
| `sweep_delta_scope.py` | Compares "narrow" data (near the goal only) with "wide" data (out to the safety boundary). |
| `safety_aware_sindy.py` | Tries fitting SINDy with extra weight near the boundary. |
| `measure_lambda_bar.py`, `theory_constants.py` | Estimate constants used in the old bound. |
| `plot_*.py` | Make the figures in `figures/2_learned_dynamics/`. |

**`sindy/`, phase 3 files:**

| File | Does |
|---|---|
| `theory_tools.py` | Computes every quantity in the theorems for the pendulum (error envelope, τ̂, τ_δ, δ*). |
| `theory_nd.py` | The same calculations for any dimension (cart-pole, two-pole). |
| `robust_certificate.py` | A learned certificate trained directly on the theorem's condition. |
| `exp_scalar.py` | 1-D sanity checks of the tightness and lower-bound results (stage 0). |
| `exp_theory.py` | Runs **one** configuration of stages 1–12 and writes one CSV row per policy. |
| `exp_nd.py` | Runs one configuration of the "sharpened" multi-system experiments. |
| `plot_theory.py` | Merges results and writes the figures and `summary.txt`. |

---

## `scripts/`

These paths assume the **original repo layout**. In the original repo, `data/<X>` and
`figures/<X>` correspond to `results/…/<X>` and `figures/…/<X>` in this export.

| Script | Runs | Output here |
|---|---|---|
| `run_sindy_pipeline.sh` | phase-2 pipeline, run once | `results/1_sindy_pipeline/` |
| `run_sweep.sh`, `run_sweep_parallel.sh` | phase-2 noise sweep | `results/2_…/noise_sweep/` |
| `run_cluster_all.sh` (+ `sync_to/from_cluster.sh`) | phase-2 sweeps on the GPU cluster `bruinml` | `results/2_…/cluster_unified_sweeps/`, `omega_rob.png` |
| `run_theory_experiments.sh` | **phase-3 stages 0–12** (stage list in its header) | `results/3_theory_validation/<run>/` |
| `run_overnight.sh` | three phase-3 variants: `theory_quad`, `theory_nn_long`, `theory_nn_nobarrier` | same |
| `run_sharpened.sh` | pendulum / cart-pole / two-pole sweeps | `results/3_…/sharpened/` |
| `progress_theory.sh` | live progress monitor | — |

---

## `docs/`

- **`PAPER_PLAN.md`**: the research plan. It lists the claims, which theorems are done,
  and the open issues (TH12–TH17). It refers to `paperv2.tex`, which is not included.
- **`wide_experiment_explained.md`**: a plain-language write-up of the main phase-2
  finding. Collecting data near the safety boundary roughly doubles the noise the
  controller tolerates. Reweighting the SINDy fit adds nothing.
- `original_repo_README.md`: the repo's original README, which describes phase 1.

---

## `results/`

### `1_sindy_pipeline/`
Output of one run of the phase-2 pipeline: `sindy_transitions.npz` (collected data),
`sindy_model.pkl` (fitted PySINDy model; loading it needs pysindy, and since it is a
pickle, only load it in a trusted environment), `sindy_model_coefficients.npz`.

### `2_learned_dynamics_sweeps/`
| Folder | Contents |
|---|---|
| `noise_sweep/` | Three versions of the noise sweep. `v0` used tiny datasets and δ blew up into the thousands, so it was abandoned. `v1` came next, and `v2_corrected` is the final one. |
| `cluster_unified_sweeps/` | The final phase-2 tables: `unified_narrow.csv`, `unified_wide.csv`, and `unified_narrow_robust.csv` (robust-loss ablation). |
| `narrow_vs_wide_data/` | `final_*.csv` holds safety vs noise for each data regime. `delta_scope_*.csv` holds box error vs reachable-set error. |
| `safety_aware_sindy/` | Baseline vs boundary-weighted SINDy (no difference found). |

Key columns: `obs_noise_std` (sensor noise σ), `delta_sup` / `delta_mean` (model
error), `safety_rate` (fraction of rollouts that never hit the unsafe region),
`goal_rate`, `limsup_V` (how close to the goal V settles), `predicted_limsup` (what the
bound predicted).

### `3_theory_validation/`
Each subfolder is **one full run** of the stage pipeline with a different certificate V:

| Run | Certificate V | Notes |
|---|---|---|
| **`theory_learned/`** | robust learned certificate | **Most complete and most recent run.** Includes the diffusion-policy stages 11–12. |
| `theory_quad/` | fixed quadratic V | Clean check of the theorems. |
| `theory/` | neural-net V | First full run. Includes the 1-D checks (`stage0_scalar/`). |
| `theory_nn_long/`, `theory_nn_nobarrier/` | neural-net V, trained longer, or with no barrier term | Tests whether a trained network satisfies the theorem's margin. |
| `theory_pilot/` | neural-net V | Small pilot run. |
| `test_launch_quad/`, `test_launch_nn/` | — | 3-job smoke tests (seed 0 only). Can be ignored. |
| `sharpened/` | — | Multi-system sweeps: `pend_*` = pendulum, `cp_*` = cart-pole, `tp_*` = two-pole; `*_noise` varies noise, `*_samples` varies data size. |

Inside each run:
- **`summary.txt`**: a plain-text pass/fail report for each theorem check. **Start here.**
- `stageK_name/results.csv`: one row per (configuration × policy).
- `stageK_name/jobs.txt`: the exact arguments of every job in that stage.
- `stageK_name/job_logs.txt`: every job's console output, joined into one file.
- `ref_v/ref_seed*.pt`, `cert/cert_seed*.pt`: the trained certificate networks (one per
  seed). Stages 1 and 9 only train these, so they have no `results.csv`.

What each stage tests:

| Stage | Varies | Checks |
|---|---|---|
| 0 `stage0_scalar` | 1-D toy problem | the bound is tight; minimax lower bound |
| 1 `stage1_ref` | — | trains a reference V |
| 2 `stage2_noise_fixed` | noise σ | final level τ̂ shrinks like δ² (Thm 2), safety (Thm 1), error envelope holds (Prop 5) |
| 3 `stage3_samples_fixed` | data size N | δ ~ N^-1/2 and τ̂ ~ 1/N (Cor 6) |
| 6 `stage6_misspec_fixed` | wrong feature library | error stops shrinking at a floor |
| 7 `stage7_eiv_fixed` | noise on the measured states | outside the theory; shows what breaks |
| 9–10 `stage9_cert`, `stage10_cert_arms` | learned certificate variants | the theorems with a learned V |
| 11–12 `stage11_diffusion`, `stage12_diffusion_matched` | diffusion policy | the theorems with the actual diffusion controller |

Key columns: `policy` (`qp`, `qp_filter`, `diffusion`, `diffusion_filter`), `sigma`,
`N`, `seed`, `delta_c_true` (true model error), `delta_c_env` (certified error bound),
`env_valid` (did the bound hold), `tau_hat` (predicted final level), `tau_delta` (Thm 2
bound), `limsup_V_mean` / `tail_max_V` (measured final level), `c` (safety level),
`safe_V_rate` / `safe_rate` (measured safety), and `thm2_holds` / `thm2_applicable`.

---

## `figures/`

| Folder / file | Shows |
|---|---|
| `2_learned_dynamics/tradeoff_sweep.png`, `corrected_tradeoff.png`, `tradeoff_salvage.png`, `tradeoff_partial.png` | model error δ vs safety / final V (successive versions) |
| `2_learned_dynamics/delta_scope.png` | why narrow-data models look accurate but are less safe |
| `2_learned_dynamics/omega_rob.png` | the region where the learned model can't guarantee decrease, for several noise levels |
| `2_learned_dynamics/clbf_comparison.png` | certificate trained on true vs SINDy physics |
| `3_theory_validation/<run>/fig_thm1.png` | measured final V vs the Theorem 1 prediction |
| `3_theory_validation/<run>/fig_rates.png` | the δ² and 1/N rates |
| `3_theory_validation/theory/fig_scalar.png` | 1-D tightness and lower-bound checks |

---

## `logs/`

- `console/`: the terminal output of each big run.
  - `train_original.log`, `train_sindy.log`: controller training (true vs SINDy physics).
    They are mostly progress-bar output.
  - `sweep_parallel.log`: phase-2 sweep.
  - `theory_cpu.log` (`theory` run), `overnight*.log` (`theory_quad` / `nn_long` /
    `nn_nobarrier`), `theory_learned*.log`, `sharpened.log`: phase-3 runs.
- `training_runs/`: TensorBoard logs and the final checkpoint
  (`checkpoints/epoch=50-step=7191.ckpt`) of the three completed controller trainings:
  `true_dynamics_run_a`, `true_dynamics_run_b` (same settings, two runs) and
  `sindy_dynamics`. View them with `tensorboard --logdir logs/training_runs`.

---

## `git_info/`

Most of this work was never committed: the repo has 3 commits and phase 3 is entirely
untracked.
- `git_log.txt`, `git_status.txt`: history and the state of the working tree.
- `uncommitted_code_changes.diff`: edits to files that are tracked in git.

---

## What was removed during cleanup

All of it can still be found in the original repo.
- Paper drafts (`paperv1.tex`, `paperv2.tex`, `empty.tex`, the compiled PDF).
- Per-job `parts/*.csv` files. Their rows were checked to be identical to the merged
  `results.csv` files. One merged file was empty (`theory_pilot` stage 2), so it was
  rebuilt from its parts.
- Duplicates: top-level copies of the theory figures, and copies of the `sharpened` CSVs.
- Old source files deleted from the repo (other robot systems), and TensorBoard runs that
  crashed on start.
- Clutter: `__pycache__`, `.DS_Store`, editor config, empty marker files.
