#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Load your API settings into the environment before running this script.
# WORKERS controls concurrency; extra CLI arguments override the defaults below.
exec "${PYTHON_BIN:-python}" main.py \
  --mode full \
  --input data/seed_problems.json \
  --problems 1800 \
  --rollouts 20 \
  --depth 10 \
  --workers "${WORKERS:-4}" \
  --progress-weight 0.25 \
  --exploration-c 1.4 \
  --output-dir output/run_r20_d10 \
  "$@"
