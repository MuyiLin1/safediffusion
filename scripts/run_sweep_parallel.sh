#!/usr/bin/env bash
# =============================================================================
# Parallel local sweep: run many (noise, seed) configs at once across CPU cores.
#
# Each config is independent, so instead of one slow serial run we launch
# PARALLEL processes at a time, each writing its own part CSV. When all configs
# finish we merge the parts into one results file.
#
# Usage (defaults shown):
#   PARALLEL=6 bash run_sweep_parallel.sh
#
# Tunables (override via environment):
#   PYTHON     python interpreter to use
#   PARALLEL   how many configs to run simultaneously (cores / ~2)
#   THREADS    torch/OMP threads PER process (keep small to avoid oversubscribe)
#   SEEDS      seeds to sweep
#   NUM_TRAJ   fixed dataset size
#   OBS_NOISE  noise grid (the delta knob)
#   MAX_EPOCHS training epochs per config
#   NUM_ROLLOUTS / T_SIM   evaluation settings
# =============================================================================
set -u

PYTHON="${PYTHON:-/Users/linmuyi/code/.venv/bin/python}"
PARALLEL="${PARALLEL:-6}"
THREADS="${THREADS:-2}"
SEEDS="${SEEDS:-0 1 2}"
NUM_TRAJ="${NUM_TRAJ:-50}"
OBS_NOISE="${OBS_NOISE:-0.0 0.02 0.05 0.1 0.15 0.2 0.3}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
T_SIM="${T_SIM:-10.0}"
PARTS_DIR="${PARTS_DIR:-data/parts}"
OUTPUT="${OUTPUT:-data/sweep_results.csv}"

mkdir -p "$PARTS_DIR"

echo "================================================================"
echo " Parallel SINDy sweep"
echo "   python     = $PYTHON"
echo "   parallel   = $PARALLEL processes x $THREADS threads each"
echo "   seeds      = $SEEDS"
echo "   noise grid = $OBS_NOISE"
echo "   max_epochs = $MAX_EPOCHS   rollouts=$NUM_ROLLOUTS  t_sim=$T_SIM"
echo "   parts dir  = $PARTS_DIR    merged -> $OUTPUT"
echo "================================================================"

# Build the job list: one line "noise seed" per config.
JOBS_FILE="$(mktemp)"
for ns in $OBS_NOISE; do
    for s in $SEEDS; do
        echo "$ns $s"
    done
done > "$JOBS_FILE"
echo "Total configs: $(wc -l < "$JOBS_FILE")"

# Run the pool. Each worker runs ONE config into its own part CSV + log.
export PYTHON THREADS NUM_TRAJ MAX_EPOCHS NUM_ROLLOUTS T_SIM PARTS_DIR
xargs -P "$PARALLEL" -n 2 sh -c '
    ns="$1"; s="$2"
    out="$PARTS_DIR/part_${ns}_${s}.csv"
    log="$PARTS_DIR/log_${ns}_${s}.log"
    echo "[start] noise=$ns seed=$s -> $out"
    OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    VECLIB_MAXIMUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    "$PYTHON" -m Lyapunov_guided_diffusion.sindy.sweep \
        --num_trajectories "$NUM_TRAJ" \
        --obs_noise_std "$ns" \
        --seeds "$s" \
        --max_epochs "$MAX_EPOCHS" \
        --accelerator cpu \
        --num_rollouts "$NUM_ROLLOUTS" \
        --t_sim "$T_SIM" \
        --output "$out" > "$log" 2>&1
    if tail -n +2 "$out" 2>/dev/null | grep -q .; then
        echo "[done ] noise=$ns seed=$s OK"
    else
        echo "[FAIL ] noise=$ns seed=$s (no row; see $log)"
    fi
' _ < "$JOBS_FILE"

rm -f "$JOBS_FILE"

# Merge all part CSVs (header once, then every data row).
echo ""
echo "=== Merging parts into $OUTPUT ==="
first_part="$(ls "$PARTS_DIR"/part_*.csv 2>/dev/null | head -1)"
if [ -z "$first_part" ]; then
    echo "No part files produced; nothing to merge."
    exit 1
fi
head -1 "$first_part" > "$OUTPUT"
for f in "$PARTS_DIR"/part_*.csv; do
    tail -n +2 "$f" 2>/dev/null >> "$OUTPUT"
done
rows=$(( $(wc -l < "$OUTPUT") - 1 ))
echo "Merged rows: $rows  ->  $OUTPUT"
echo "Done."
