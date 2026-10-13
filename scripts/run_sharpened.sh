#!/usr/bin/env bash
# Sharpened-theory experiments (pendulum + cart-pole) with exp_nd.py.
# Usage: WORKERS=13 bash run_sharpened.sh ; results in data/sharpened/<sweep>.csv
set -u
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/Users/linmuyi/code/.venv/bin/python}"
WORKERS="${WORKERS:-12}"
ROOT="${ROOT:-data/sharpened}"
SEEDS="${SEEDS:-0 1 2 3 4}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON ROOT

J=$(mktemp)
for s in $SEEDS; do
    for sg in 0 0.1 0.2 0.5 1; do
        echo "pend_noise pend_quad_sig${sg}_s$s --system pendulum --cert quad --sigma $sg --num_trajectories 50 --seed $s"
        echo "pend_noise pend_net_sig${sg}_s$s --system pendulum --cert net --sigma $sg --num_trajectories 50 --seed $s"
    done
    for nt in 5 10 20 50 100 200; do
        echo "pend_samples pend_quad_n${nt}_s$s --system pendulum --cert quad --sigma 0.5 --num_trajectories $nt --seed $s"
    done
    for sg in 0 0.05 0.1 0.2 0.5; do
        echo "cp_noise cp_quad_sig${sg}_s$s --system cartpole --cert quad --sigma $sg --num_trajectories 200 --seed $s"
    done
    for nt in 25 50 100 200 400 800; do
        echo "cp_samples cp_quad_n${nt}_s$s --system cartpole --cert quad --sigma 0.1 --num_trajectories $nt --seed $s"
    done
    for sg in 0 0.1; do
        echo "cp_res cp_res_sig${sg}_s$s --system cartpole --cert res --sigma $sg --num_trajectories 200 --seed $s"
    done
    for sg in 0 0.005 0.01 0.02 0.05; do
        echo "tp_noise tp_quad_sig${sg}_s$s --system twopole --cert quad --sigma $sg --num_trajectories 1600 --rho 1e-8 --seed $s"
    done
done > "$J"
echo "$(wc -l < "$J") jobs, $WORKERS workers ($(date))"

run_one() {
    set -- $1
    local sweep=$1 id=$2; shift 2
    mkdir -p "$ROOT/$sweep/parts" "$ROOT/$sweep/logs"
    local out="$ROOT/$sweep/parts/$id.csv"
    [[ -s "$out" ]] && return 0
    if "$PYTHON" -m Lyapunov_guided_diffusion.sindy.exp_nd "$@" --output "$out.tmp" \
        > "$ROOT/$sweep/logs/$id.log" 2>&1 && [[ -s "$out.tmp" ]]; then
        mv "$out.tmp" "$out"; echo "[done] $sweep/$id"
    else
        echo "[FAIL] $sweep/$id"
    fi
}
export -f run_one
tr '\n' '\0' < "$J" | xargs -0 -P "$WORKERS" -n 1 bash -c 'run_one "$1"' _
rm -f "$J"
for d in "$ROOT"/*/; do
    "$PYTHON" -m Lyapunov_guided_diffusion.sindy.plot_theory merge "$d/parts" "${d%/}.csv"
done
echo "ALL SHARPENED RUNS FINISHED $(date)"
