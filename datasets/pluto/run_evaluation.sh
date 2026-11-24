#!/usr/bin/env bash
set -euo pipefail
pluto_task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pluto_study_root="${PLUTO_STUDY_ROOT:-$HOME/Wage_surrogate}"
pluto_repo="${PLUTO_GENWP_REPO:-$pluto_study_root/GenWP-pub}"
pluto_bench="${PLUTO_BENCHMARK:-$pluto_study_root/mcwp_ablation/benchmark_all_accuracy_runtime.py}"
pluto_cases="${PLUTO_CASES_DIR:-$pluto_repo/dataset/pluto_ood_outputs}"
pluto_out="${PLUTO_EVAL_OUT:-$pluto_study_root/results/pluto_ood_$(date +%Y%m%d_%H%M%S)}"
pluto_python="${PLUTO_PYTHON:-python}"

pluto_base="$pluto_study_root/mcwp/runs/base/eval_checkpoints/ema_100000.pt"
pluto_long="$pluto_study_root/mcwp_nccshift_v3/runs/long100_mseonly_ft5k/eval_checkpoints/ema_004000.pt"
pluto_authors="$pluto_repo/genwp/checkpoints/trained_model.pt"
pluto_retrained="$pluto_repo/experiments/genwp_ddp_20k/checkpoints/ema_0.999_100000.pt"
pluto_fno="$pluto_study_root/fno_baseline/runs/fno_l_200test_workers3/eval_checkpoints/ema_100000.pt"
for pluto_checkpoint in "$pluto_base" "$pluto_long" "$pluto_authors" "$pluto_retrained"; do
    if [[ ! -f "$pluto_checkpoint" ]]; then
        printf 'Missing study checkpoint: %s\n' "$pluto_checkpoint" >&2
        exit 1
    fi
done
"$pluto_python" "$pluto_task_dir/benchmark_pluto.py" \
    --benchmark "$pluto_bench" --cases-dir "$pluto_cases" --check-only

pluto_options=(
    --repo "$pluto_repo" --out-dir "$pluto_out" --device cuda:0
    --warmups 1 --repeats 5 --seed 5100 --fd-sanity first
    --fd-dx 10 --fd-dz 10 --fd-dt 0.001 --fd-nt 1001
    --fd-freq 15 --fd-pml 50 --fd-downsample 10
    --mcwp Base_100k "$pluto_base"
    --mcwp Proposed_Long100_MSE_4k "$pluto_long"
    --genwp GenWP_Authors "$pluto_authors"
    --genwp GenWP_Retrained_100k "$pluto_retrained"
)
# Reuse the study's FNO loader if this CURRENT benchmark exposes it. Do not
# invent an FNO implementation or silently treat a reduced run as five models.
pluto_help="$("$pluto_python" "$pluto_bench" --help)"
if [[ "$pluto_help" == *"  --fno "* || "$pluto_help" == *"[--fno "* ]]; then
    if [[ ! -f "$pluto_fno" ]]; then
        printf 'FNO flag exists, but checkpoint is missing: %s\n' "$pluto_fno" >&2
        exit 1
    fi
    pluto_options+=(--fno FNO_100k "$pluto_fno")
else
    printf '%s\n' 'This benchmark does not expose --fno: evaluating Base, Long100 and both GenWP weights. Use the current FNO-capable study benchmark for the five-model comparison.' >&2
fi
"$pluto_python" "$pluto_task_dir/benchmark_pluto.py" \
    --benchmark "$pluto_bench" --cases-dir "$pluto_cases" "${pluto_options[@]}"
printf 'Pluto results: %s\n' "$pluto_out"
