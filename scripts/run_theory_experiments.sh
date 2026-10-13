#!/usr/bin/env bash
# =============================================================================
# Theory-validation experiments for paperv2.tex. Stages run one after another;
# inside a stage, jobs run in parallel on WORKERS workers. Re-running the script
# skips every job that already finished (resume after a crash or a logout).
#
#   Stage 0  1-D checks of Prop 3 (tightness) and Prop 7 (lower bound)       CPU, seconds
#   Stage 1  train one reference certificate V per seed on clean data          GPU/CPU, training
#   Stage 2  noise sweep, V fixed            -> Thm 2 (tau_hat = O(delta^2)), Thm 1, Prop 5
#   Stage 3  sample-size sweep, V fixed      -> Cor 6 (delta ~ N^-1/2, tau_hat ~ 1/N)
#   Stage 4  noise sweep, V retrained        -> delta_star vs the measured safety cliff
#   Stage 5  narrow vs wide data, retrained  -> Cor 4 (coverage of the top band)
#   Stage 6  misspecified library, V fixed   -> Prop 5 floor m_bar (error stops shrinking)
#   Stage 7  noisy states (outside theory)   -> what breaks when Prop 5's model is violated
#   Stage 8  merge + figures + summary.txt
#
# Usage (on the cluster, from the repo root):
#   WORKERS=12 nohup bash run_theory_experiments.sh > theory_run.log 2>&1 &
#   bash progress_theory.sh                    # one-shot status
#   watch -n 30 bash progress_theory.sh        # live status
#
# Knobs (environment variables):
#   WORKERS       parallel jobs per stage                         (default 8)
#   TRAIN_WORKERS parallel jobs in the training stages 1,4,5      (default WORKERS)
#   THREADS       CPU threads per job                             (default 2)
#   ACCELERATOR   gpu | cpu for the training stages 1,4,5         (default gpu, falls back to cpu)
#   GPU_IDS       GPUs to round-robin over, e.g. "0 1 2 3"        (default: all visible)
#   STAGES        which stages to run, e.g. "2 3 8"               (default "0 1 2 3 4 5 6 7 8")
#   SEEDS         seeds                                           (default "0 1 2 3 4")
#   POLICIES      policies evaluated per config                   (default all four)
#   LIBRARY       feature library for f_hat: bias | exact | rich  (default bias)
#   PYTHON        interpreter                                     (default ~/safediffusion/.venv/bin/python)
#   ROOT          output root                                     (default data/theory)
#   REF_EPOCHS / REF_TRAJ / RETRAIN_EPOCHS / NUM_ROLLOUTS / T_SIM
#   SIGMAS_FIXED / NTRAJ_GRID / SIGMA_N / SIGMAS_RETRAIN / SIGMAS_COVER /
#   NTRAJ_MISSPEC / SIGMA_MISSPEC / SIGMAS_EIV                    per-stage grids
# =============================================================================
set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-$HOME/safediffusion/.venv/bin/python}"
WORKERS="${WORKERS:-8}"
TRAIN_WORKERS="${TRAIN_WORKERS:-$WORKERS}"
THREADS="${THREADS:-2}"
ACCELERATOR="${ACCELERATOR:-gpu}"
GPU_IDS="${GPU_IDS:-auto}"
STAGES="${STAGES:-0 1 2 3 4 5 6 7 8}"
SEEDS="${SEEDS:-0 1 2 3 4}"
POLICIES="${POLICIES:-qp qp_filter diffusion diffusion_filter}"
LIBRARY="${LIBRARY:-bias}"
CERT="${CERT:-nn}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"
CERT_DIR="${CERT_DIR:-${ROOT:-data/theory}/cert}"
CERT_SIGMA="${CERT_SIGMA:-0.5}"
CERT_NTRAJ="${CERT_NTRAJ:-50}"
CERT_STEPS="${CERT_STEPS:-3000}"
SIGMAS_C3="${SIGMAS_C3:-0 0.2 0.5 1 2}"
SIGMAS_DIFF="${SIGMAS_DIFF:-0 0.5 1}"
DIFF_SEEDS="${DIFF_SEEDS:-0 1 2}"
DIFF_ROLLOUTS="${DIFF_ROLLOUTS:-10}"
SHIFT_GOAL="${SHIFT_GOAL:-0}"
U_MAX="${U_MAX:-30}"
REF_DIR="${REF_DIR:-${ROOT:-data/theory}/ref_v}"
FIG_DIR="${FIG_DIR:-figures/$(basename "${ROOT:-data/theory}")}"
ROOT="${ROOT:-data/theory}"
REF_EPOCHS="${REF_EPOCHS:-10}"
REF_TRAJ="${REF_TRAJ:-200}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-5}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-30}"
T_SIM="${T_SIM:-10}"
SIGMAS_FIXED="${SIGMAS_FIXED:-0 0.1 0.2 0.5 1 2 4}"
NTRAJ_GRID="${NTRAJ_GRID:-2 5 10 20 50 100 200}"
SIGMA_N="${SIGMA_N:-0.5}"
SIGMAS_RETRAIN="${SIGMAS_RETRAIN:-0 0.1 0.2 0.5 1 2 4}"
SIGMAS_COVER="${SIGMAS_COVER:-0.1 0.5 1 2}"
NTRAJ_MISSPEC="${NTRAJ_MISSPEC:-5 20 50 100 200 500}"
SIGMA_MISSPEC="${SIGMA_MISSPEC:-0.2}"
SIGMAS_EIV="${SIGMAS_EIV:-0.02 0.05 0.1}"

MOD="Lyapunov_guided_diffusion.sindy"
mkdir -p "$ROOT"
echo "$(date +%s)" > "$ROOT/.started"

# ---- GPU detection ----------------------------------------------------------
if [[ "$ACCELERATOR" == "gpu" ]]; then
    NGPU=$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "$PYTHON" -c \
        "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
    if [[ "$NGPU" == "0" ]]; then
        echo "WARNING: no usable GPU found by torch, training stages fall back to CPU"
        ACCELERATOR="cpu"
    elif [[ "$GPU_IDS" == "auto" ]]; then
        GPU_IDS="$(seq -s ' ' 0 $((NGPU - 1)))"
    fi
fi
[[ "$ACCELERATOR" == "cpu" ]] && GPU_IDS=""

echo "================================================================"
echo " theory experiments   $(date)"
echo "   python      $PYTHON"
echo "   workers     $WORKERS x $THREADS threads"
echo "   accelerator $ACCELERATOR   gpus: ${GPU_IDS:-none}"
echo "   stages      $STAGES"
echo "   seeds       $SEEDS"
echo "   policies    $POLICIES"
echo "   output      $ROOT"
echo "================================================================"

# ---- one job ----------------------------------------------------------------
# A job line is: "<index> <job_id> <uses_gpu 0|1> <exp_theory args...>"
run_job() {
    set -f
    # shellcheck disable=SC2086
    set -- $1
    set +f
    local idx="$1" id="$2" gpu="$3"
    shift 3
    local dir="$ROOT/$STAGE_NAME"
    local out="$dir/parts/$id.csv" log="$dir/logs/$id.log"
    if [[ -e "$dir/parts/$id.done" ]]; then
        return 0
    fi
    rm -f "$dir/parts/$id.failed"
    local cuda="-1" acc="cpu"
    if [[ "$gpu" == "1" && -n "$GPU_IDS" ]]; then
        local arr=($GPU_IDS)
        cuda="${arr[$((idx % ${#arr[@]}))]}"
        acc="gpu"
    fi
    echo "[start $(date +%H:%M:%S)] $STAGE_NAME/$id (cuda=$cuda)"
    local t0=$SECONDS
    # train_ref writes the --ref_v file; every other mode writes the CSV.
    local expect="$out" prev="" a
    if [[ " $* " == *" train_ref "* ]]; then
        for a in "$@"; do [[ "$prev" == "--ref_v" ]] && expect="$a"; prev="$a"; done
    fi
    if [[ " $* " == *" train_cert "* ]]; then
        for a in "$@"; do [[ "$prev" == "--cert_file" ]] && expect="$a"; prev="$a"; done
    fi
    rm -f "$expect"
    # Lightning catches SIGTERM and exits 0, so success also requires the output file.
    if CUDA_VISIBLE_DEVICES="$cuda" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        OPENBLAS_NUM_THREADS="$THREADS" PYTHONUNBUFFERED=1 \
        "$PYTHON" -m "$MOD.exp_theory" "$@" --accelerator "$acc" --output "$out" \
        > "$log" 2>&1 && [[ -s "$expect" ]]; then
        touch "$dir/parts/$id.done"
        echo "[done  $(date +%H:%M:%S)] $STAGE_NAME/$id ($((SECONDS - t0))s)"
    else
        touch "$dir/parts/$id.failed"
        echo "[FAIL  $(date +%H:%M:%S)] $STAGE_NAME/$id, see $log"
    fi
}
export -f run_job
export PYTHON THREADS ROOT MOD GPU_IDS

# ---- one stage --------------------------------------------------------------
run_stage() {
    local name="$1" jobs_file="$2" nw="${3:-$WORKERS}"
    STAGE_NAME="$name"
    export STAGE_NAME
    mkdir -p "$ROOT/$name/parts" "$ROOT/$name/logs"
    awk '{print NR-1, $0}' "$jobs_file" > "$ROOT/$name/jobs.txt"
    local total
    total=$(wc -l < "$ROOT/$name/jobs.txt")
    echo
    echo "=== $name: $total jobs, $nw workers ($(date)) ==="
    tr '\n' '\0' < "$ROOT/$name/jobs.txt" | xargs -0 -P "$nw" -n 1 bash -c 'run_job "$1"' _
    "$PYTHON" -m "$MOD.plot_theory" merge "$ROOT/$name/parts" "$ROOT/$name.csv"
    local nf
    nf=$(find "$ROOT/$name/parts" -name '*.failed' | wc -l)
    echo "=== $name finished: $((total - nf)) ok, $nf failed ($(date)) ==="
    rm -f "$jobs_file"
}

want() { [[ " $STAGES " == *" $1 "* ]]; }
COMMON="--num_rollouts $NUM_ROLLOUTS --t_sim $T_SIM --policies $POLICIES --cert $CERT --u_max $U_MAX"
[[ "$SHIFT_GOAL" == "1" ]] && COMMON="$COMMON --shift_goal"
J=$(mktemp)

# ---- stage 0 ----------------------------------------------------------------
if want 0; then
    echo; echo "=== stage0_scalar ($(date)) ==="
    mkdir -p "$ROOT/stage0_scalar"
    "$PYTHON" -m "$MOD.exp_scalar" --out_dir "$ROOT/stage0_scalar" \
        > "$ROOT/stage0_scalar/run.log" 2>&1 \
        && echo "=== stage0_scalar finished ===" \
        || echo "=== stage0_scalar FAILED, see $ROOT/stage0_scalar/run.log ==="
fi

# ---- stage 1: reference certificates ----------------------------------------
if want 1; then
    : > "$J"
    for s in $SEEDS; do
        echo "ref_s$s 1 --mode train_ref --ref_v $REF_DIR/ref_seed$s.pt --seed $s --num_trajectories $REF_TRAJ --max_epochs $REF_EPOCHS $TRAIN_EXTRA" >> "$J"
    done
    mkdir -p "$REF_DIR"
    run_stage stage1_ref "$J" "$TRAIN_WORKERS"
fi

ref() { echo "--ref_v $REF_DIR/ref_seed$1.pt --cert_file $CERT_DIR/cert_seed$1.pt"; }

# ---- stage 9: learned certificates (robust training, one per seed) -----------
if want 9; then
    : > "$J"
    for s in $SEEDS; do
        echo "cert_s$s 0 --mode train_cert --cert_file $CERT_DIR/cert_seed$s.pt --library $LIBRARY --sigma $CERT_SIGMA --seed $s --num_trajectories $CERT_NTRAJ --cert_steps $CERT_STEPS" >> "$J"
    done
    mkdir -p "$CERT_DIR"
    run_stage stage9_cert "$J"
fi

# ---- stage 2: noise sweep, fixed V ------------------------------------------
if want 2; then
    : > "$J"
    for sg in $SIGMAS_FIXED; do for s in $SEEDS; do
        echo "sig${sg}_s$s 0 --mode fixed $(ref $s) --library $LIBRARY --sigma $sg --seed $s --num_trajectories 50 $COMMON" >> "$J"
    done; done
    run_stage stage2_noise_fixed "$J"
fi

# ---- stage 3: sample-size sweep, fixed V ------------------------------------
if want 3; then
    : > "$J"
    for nt in $NTRAJ_GRID; do for s in $SEEDS; do
        echo "n${nt}_s$s 0 --mode fixed $(ref $s) --library $LIBRARY --sigma $SIGMA_N --seed $s --num_trajectories $nt $COMMON" >> "$J"
    done; done
    run_stage stage3_samples_fixed "$J"
fi

# ---- stage 4: noise sweep, retrained V --------------------------------------
if want 4; then
    : > "$J"
    for sg in $SIGMAS_RETRAIN; do for s in $SEEDS; do
        echo "sig${sg}_s$s 1 --mode retrain --library $LIBRARY --sigma $sg --seed $s --num_trajectories 50 --max_epochs $RETRAIN_EPOCHS $TRAIN_EXTRA $COMMON" >> "$J"
    done; done
    run_stage stage4_noise_retrain "$J" "$TRAIN_WORKERS"
fi

# ---- stage 5: narrow vs wide data, retrained V ------------------------------
if want 5; then
    : > "$J"
    for icr in 0.0 1.4; do for sg in $SIGMAS_COVER; do for s in $SEEDS; do
        echo "ic${icr}_sig${sg}_s$s 1 --mode retrain --library $LIBRARY --ic_radius $icr --sigma $sg --seed $s --num_trajectories 50 --max_epochs $RETRAIN_EPOCHS $TRAIN_EXTRA $COMMON" >> "$J"
    done; done; done
    run_stage stage5_coverage_retrain "$J" "$TRAIN_WORKERS"
fi

# ---- stage 6: misspecified library, fixed V ---------------------------------
if want 6; then
    : > "$J"
    for lib in $LIBRARY linear; do for nt in $NTRAJ_MISSPEC; do for s in $SEEDS; do
        echo "${lib}_n${nt}_s$s 0 --mode fixed $(ref $s) --library $lib --sigma $SIGMA_MISSPEC --seed $s --num_trajectories $nt $COMMON" >> "$J"
    done; done; done
    run_stage stage6_misspec_fixed "$J"
fi

# ---- stage 7: noisy states (errors in variables), fixed V --------------------
if want 7; then
    : > "$J"
    for sg in $SIGMAS_EIV; do for s in $SEEDS; do
        echo "eiv${sg}_s$s 0 --mode fixed $(ref $s) --library $LIBRARY --noise_on both --sigma $sg --sigma_x $sg --seed $s --num_trajectories 50 $COMMON" >> "$J"
    done; done
    run_stage stage7_eiv_fixed "$J"
fi

# ---- stage 10: nominal vs robust certificate training, per configuration -----
if want 10; then
    : > "$J"
    for sg in $SIGMAS_C3; do for s in $SEEDS; do for arm in learned_nominal learned_robust; do
        echo "${arm}_sig${sg}_s$s 0 --mode fixed --library $LIBRARY --sigma $sg --seed $s --num_trajectories 50 $COMMON --cert $arm --cert_steps $CERT_STEPS" >> "$J"
    done; done; done
    run_stage stage10_cert_arms "$J"
fi

# ---- stage 11: diffusion sampler guided by the learned certificate ------------
if want 11; then
    : > "$J"
    for sg in $SIGMAS_DIFF; do for s in $DIFF_SEEDS; do
        echo "diff_sig${sg}_s$s 0 --mode fixed $(ref $s) --library $LIBRARY --sigma $sg --seed $s --num_trajectories 50 $COMMON --cert learned --policies diffusion diffusion_filter --num_rollouts $DIFF_ROLLOUTS" >> "$J"
    done; done
    run_stage stage11_diffusion "$J"
fi

# ---- stage 12: as stage 11, with the sampler's control set matched to U ------
if want 12; then
    : > "$J"
    for sg in $SIGMAS_DIFF; do for s in $DIFF_SEEDS; do
        echo "diffm_sig${sg}_s$s 0 --mode fixed $(ref $s) --library $LIBRARY --sigma $sg --seed $s --num_trajectories 50 $COMMON --cert learned --policies diffusion diffusion_filter --num_rollouts $DIFF_ROLLOUTS --match_u_limits" >> "$J"
    done; done
    run_stage stage12_diffusion_matched "$J"
fi

# ---- stage 8: report ----------------------------------------------------------
if want 8; then
    echo; echo "=== stage8_report ($(date)) ==="
    "$PYTHON" -m "$MOD.plot_theory" report --root "$ROOT" --fig_dir "$FIG_DIR"
fi
rm -f "$J"
echo; echo "ALL REQUESTED STAGES FINISHED $(date)"
