#!/usr/bin/env bash
# Overnight batch (CPU only). Three experiments, each in its own folder:
#   A  data/theory_quad          fixed quadratic certificate, stages 2 3 6 7 (fast, clean theorem test)
#   B  data/theory_nn_long       new networks: barrier + bowl, 20 epochs, then stage 2
#   C  data/theory_nn_nobarrier  new networks: bowl only, no barrier, 10 epochs, then stage 2
# B and C check whether a trained network can satisfy the theorem (lambda_bar > 0).
# Usage: caffeinate -i nohup bash run_overnight.sh > overnight.log 2>&1 &
set -u
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-/Users/linmuyi/code/.venv/bin/python}"
export ACCELERATOR=cpu THREADS=1 POLICIES="qp qp_filter"

echo "### A: quadratic certificate ($(date))"
ROOT=data/theory_quad REF_DIR=data/theory/ref_v CERT=quad U_MAX=30 \
    STAGES="2 3 6 7 8" WORKERS=4 bash run_theory_experiments.sh

echo "### B and C: train new networks in parallel ($(date))"
ROOT=data/theory_nn_long CERT=nn SHIFT_GOAL=1 U_MAX=300 REF_EPOCHS=20 \
    STAGES="1 2 8" WORKERS=5 TRAIN_WORKERS=5 bash run_theory_experiments.sh > overnight_B.log 2>&1 &
ROOT=data/theory_nn_nobarrier CERT=nn SHIFT_GOAL=1 U_MAX=300 REF_EPOCHS=10 TRAIN_EXTRA="--no_barrier" \
    STAGES="1 2 8" WORKERS=5 TRAIN_WORKERS=5 bash run_theory_experiments.sh > overnight_C.log 2>&1 &
wait
echo "### ALL OVERNIGHT RUNS FINISHED ($(date))"
