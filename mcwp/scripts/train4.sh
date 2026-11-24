#!/usr/bin/env bash
# Usage: bash scripts/train4.sh base|physics MAX_STEPS RUN_NAME auto|fresh [extra train.py options]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VARIANT=${1:?Specify base or physics}
MAX_STEPS=${2:-20000}
RUN_NAME=${3:-$VARIANT}
RESUME=${4:-auto}
if (( $# >= 4 )); then shift 4; else set --; fi
GPUS=${GPUS:-4,5,6,7}
IFS=, read -r -a GPU_IDS <<< "$GPUS"
NGPU=${#GPU_IDS[@]}
REPO=${GENWP_REPO:-$HOME/Wage_surrogate/GenWP-pub}
DATA=${DATA_DIR:-$REPO/dataset/phaseA_outputs/h5}
RUN_DIR=$ROOT/runs/$RUN_NAME
mkdir -p "$RUN_DIR"
cd "$ROOT"
printf '\nVariant=%s | GPUs=%s | Run=%s | Target=%s | Resume=%s\n' "$VARIANT" "$GPUS" "$RUN_DIR" "$MAX_STEPS" "$RESUME"
export CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
python -u -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "$ROOT/train.py" \
    --variant "$VARIANT" --data-dir "$DATA" --run-dir "$RUN_DIR" \
    --max-steps "$MAX_STEPS" --global-batch 36 --workers "${WORKERS:-2}" \
    --width "${WIDTH:-32}" --horizon "${HORIZON:-3}" --seed "${SEED:-20260909}" \
    --lr 1e-4 --ema 0.999 --log-every 100 --save-every 2000 \
    --eval-every 5000 --eval-at 1000 --keep-last 2 --preserve-full 20000,50000 \
    --resume "$RESUME" "$@" 2>&1 | tee -a "$RUN_DIR/console.log"
