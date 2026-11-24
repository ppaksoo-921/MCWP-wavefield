#!/usr/bin/env bash
# Run only after GPU EVAL_GPU is available. Default=7 (also used by train4.sh).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STEP=${1:-20000}
FORMATTED=$(printf '%06d' "$STEP")
REPO=${GENWP_REPO:-$HOME/Wage_surrogate/GenWP-pub}
OUT=$ROOT/evaluation/compare_${FORMATTED}_$(date +%Y%m%d_%H%M%S)
cd "$ROOT"
CUDA_VISIBLE_DEVICES=${EVAL_GPU:-7} python -u evaluate.py \
    --checkpoint "$ROOT/runs/base/eval_checkpoints/ema_${FORMATTED}.pt" \
    --checkpoint "$ROOT/runs/physics/eval_checkpoints/ema_${FORMATTED}.pt" \
    --include-official --repo "$REPO" --out-dir "$OUT" --seeds 0 --save-predictions
printf '\nResults: %s\n' "$OUT"
