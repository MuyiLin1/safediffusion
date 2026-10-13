#!/usr/bin/env bash
set -e
REMOTE_DIR="code/safediffusion"

echo "=== Syncing results back from bruinml:$REMOTE_DIR ==="
mkdir -p data logs

rsync -avz bruinml:$REMOTE_DIR/data/unified_*.csv ./data/ 2>/dev/null || echo "No unified CSVs yet"
rsync -avz bruinml:$REMOTE_DIR/omega_rob.png ./ 2>/dev/null || echo "No omega_rob.png yet"
rsync -avz bruinml:$REMOTE_DIR/logs/lambda_bar_result.txt ./logs/ 2>/dev/null || echo "No lambda_bar_result.txt yet"

echo "Sync from cluster finished."
