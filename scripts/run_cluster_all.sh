#!/usr/bin/env bash
# =============================================================================
# Master GPU Cluster Runner for All Open Items
#
# Runs on bruinml GPU cluster:
#   1. Unified Narrow Sweep (Table 1 + Table 2 narrow, 45 configs) -> data/unified_narrow.csv
#   2. Unified Wide Sweep (Table 2 wide, 25 configs) -> data/unified_wide.csv
#   3. OPEN 6 Empirical Lambda-bar measurement -> logs/lambda_bar_result.txt
#   4. OPEN 5/2 Omega_rob visualization -> omega_rob.png
#   5. OPEN 5/3 Robust Loss ablation sweep -> data/unified_narrow_robust.csv
# =============================================================================
set -e

PYTHON="${PYTHON:-python}"
ACCELERATOR="${ACCELERATOR:-gpu}"

echo "================================================================"
echo " Starting Master GPU Cluster Pipeline"
echo "   Python:      $PYTHON"
echo "   Accelerator: $ACCELERATOR"
echo "   Date:        $(date)"
echo "================================================================"

mkdir -p data logs

# Step 1: Unified Narrow Sweep
echo ""
echo ">>> [1/5] Running Unified Narrow Sweep (45 configs)..."
$PYTHON -m Lyapunov_guided_diffusion.sindy.sweep \
    --num_trajectories 50 \
    --obs_noise_std 0.0 0.01 0.02 0.03 0.04 0.05 0.075 0.10 0.15 \
    --seeds 0 1 2 3 4 \
    --max_epochs 5 \
    --accelerator "$ACCELERATOR" \
    --num_rollouts 30 \
    --ic_radius 0.0 \
    --output data/unified_narrow.csv

# Step 2: Unified Wide Sweep
echo ""
echo ">>> [2/5] Running Unified Wide Sweep (25 configs)..."
$PYTHON -m Lyapunov_guided_diffusion.sindy.sweep \
    --num_trajectories 50 \
    --obs_noise_std 0.04 0.05 0.075 0.10 0.15 \
    --seeds 0 1 2 3 4 \
    --max_epochs 5 \
    --accelerator "$ACCELERATOR" \
    --num_rollouts 30 \
    --ic_radius 1.4 \
    --output data/unified_wide.csv

# Step 3: OPEN 6 Lambda-bar measurement
echo ""
echo ">>> [3/5] Running OPEN 6 Lambda-bar Measurement..."
if [ -f "logs/inverted_pendulum_sindy/commit_b6d4525/version_3/checkpoints/epoch=50-step=7191.ckpt" ]; then
    $PYTHON -m Lyapunov_guided_diffusion.sindy.measure_lambda_bar \
        --checkpoint logs/inverted_pendulum_sindy/commit_b6d4525/version_3/checkpoints/epoch=50-step=7191.ckpt \
        --num_rollouts 30 | tee logs/lambda_bar_result.txt
else
    echo "Notice: 51-epoch checkpoint not found at expected path. Skipping lambda_bar measurement."
fi

# Step 4: OPEN 5/2 Omega_rob visualization
echo ""
echo ">>> [4/5] Generating Omega_rob Visualization..."
$PYTHON -m Lyapunov_guided_diffusion.sindy.plot_omega_rob \
    --csv data/unified_narrow.csv \
    --sigmas 0.0 0.04 0.05 \
    --output omega_rob.png \
    --accelerator "$ACCELERATOR"

# Step 5: OPEN 5/3 Robust loss ablation sweep
echo ""
echo ">>> [5/5] Running Robust Loss Ablation Sweep (25 configs)..."
$PYTHON -m Lyapunov_guided_diffusion.sindy.sweep \
    --num_trajectories 50 \
    --obs_noise_std 0.04 0.05 0.075 0.10 0.15 \
    --seeds 0 1 2 3 4 \
    --max_epochs 5 \
    --accelerator "$ACCELERATOR" \
    --num_rollouts 30 \
    --ic_radius 0.0 \
    --robust_delta 10.0 \
    --output data/unified_narrow_robust.csv

echo ""
echo "================================================================"
echo " All cluster runs complete!"
echo " Results generated:"
echo "   - data/unified_narrow.csv"
echo "   - data/unified_wide.csv"
echo "   - data/unified_narrow_robust.csv"
echo "   - omega_rob.png"
echo "   - logs/lambda_bar_result.txt"
echo "================================================================"
