#!/usr/bin/env bash
# Sync the code (and scan.jsonl.gz, once) to Hypatia and run 'uv sync' there (login node is fine for uv sync).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${RATLAS_REMOTE:-hypatia}"
SCAN="${RATLAS_SCAN:-/Users/clem/Projects/Papers/RQ/docs/etudes_2026-10-16/scan.jsonl.gz}"
ssh "$REMOTE" 'mkdir -p ~/renascor-atlas ~/renascor-atlas-data/raw ~/renascor-atlas-data/out'
nice -n 19 rsync -a --delete --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache \
  --exclude '*.pyc' --exclude out "$REPO/" "$REMOTE:renascor-atlas/"
if [ -f "$SCAN" ]; then
  nice -n 19 rsync -a --partial "$SCAN" "$REMOTE:renascor-atlas-data/scan.jsonl.gz"
fi
ssh "$REMOTE" 'cd ~/renascor-atlas && nice -n 19 ~/.local/bin/uv sync 2>&1 | tail -3'
