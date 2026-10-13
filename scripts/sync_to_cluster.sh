#!/usr/bin/env bash
set -e
REMOTE_DIR="code/safediffusion"

echo "=== Syncing code to bruinml:$REMOTE_DIR ==="
rsync -avz \
    --rsync-path="mkdir -p $REMOTE_DIR && rsync" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'data/parts' \
    ./ bruinml:$REMOTE_DIR/

echo "Sync to cluster finished successfully."
