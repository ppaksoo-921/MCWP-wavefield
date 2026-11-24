# User guide

## Environment

The study used Python 3.10.16, PyTorch 1.12.1, NumPy 1.26.4, CUDA on an RTX
3090, FP32, and TF32 disabled. `requirements.txt` lists other Python packages.
Start from the original GenWP environment or install a CUDA/PyTorch build
compatible with your driver. Download the official GenWP repository to
`GenWP-pub/` beside the local study folders; install NeuralOperator at commit
`8a0f5260983efffbe6e8ec39c2d095dba8cb2505` to evaluate the FNO model.
The GenWP repository license and version must be retained separately.

## Source files and expected shapes

- MCWP input: velocity `vp` of shape `[128,128]` in m/s, newest-first history
  of five pressure fields, and the next physical frame index `1..100`.
- FD case NPZ: `vp` float32 `[128,128]`, `snaps` float32 `[101,128,128]`,
  `loc_x` scalar integer. Official GenWP MAT files can have 501 frames and are
  subsampled by five by `scripts/evaluate_final66.py`.
- HDF5 training file: `snaps` `[101,128,128]`, `vp` `[128,128]`, `loc_x`
  scalar. No future reference fields are fed back into free-running inference.
- Model output: 100 recursively predicted pressure fields; the reference
  initial frame is excluded from accuracy metrics.

## Prepare a geoscientific input and FD reference

The four supplied Red Sea inputs are small enough to check without CUDA:

```bash
python datasets/redsea/prepare_redsea.py
```

For the first velocity model's five sources on a machine with Torch/CUDA:

```bash
python datasets/redsea/generate_redsea_cases.py \
  --models 1 --out-dir GenWP-pub/dataset/redsea_ood_range_matched/outputs \
  --device cuda:0
```

The solver refuses to overwrite an existing case. It uses dx=dz=10 m,
dt=0.001 s, nt=1001, 15 Hz Ricker, 50 PML cells, and every tenth snapshot.
Full generation uses the default four models and writes 20 case NPZ files.
There is no pressure-amplitude rescaling between FD and model evaluation.

`datasets/kimberlina/` and `datasets/pluto/` retain their source preprocessing
and case builders. The historical Kimberlina script expects the external source
at `~/Wage_surrogate/GenWP-pub/dataset/kimberlina_ood_year0/raw/vp_year0.zip`.
The Pluto builder expects the original `int_depth_vp.sgy` and accepts `--repo`
and `--out-dir`. Consult their `--help` and source manifest before running.

## Check and re-evaluate cases

```bash
python scripts/evaluate_final66.py --groups RedSea --list
python scripts/evaluate_final66.py --data-root GenWP-pub \
  --groups RedSea --verify-inputs-only
python scripts/evaluate_final66.py --data-root GenWP-pub \
  --groups RedSea --models long100 --device cuda:0 \
  --output-dir outputs/redsea_accuracy
```

`--groups` and `--models` are repeatable. Valid model labels: `base`,
`long100`, `fno`, `genwp_published`, `genwp_retrained`, `monolithic`, and
`static_fusion`. The default is `long100`. Use `--max-cases 1` for a quick
functional run, but interpret its output as one case only. Use
`--marmousi-cases PATH` for the final 15 direct-source NPZ cases and
`--fno-source PATH` for the pinned NeuralOperator checkout. The published
GenWP and retrained GenWP paths are supplied with `--genwp-published PATH`
and `--genwp-retrained PATH`.

The runner checks every selected case file's SHA256 against the final catalog
before inference. `--verify-inputs-only` performs this check without importing
PyTorch. Outputs are `per_case.csv` and `summary.csv`. The summary includes an
equal-six-group aggregate only when all 66 cases are evaluated. It is an
accuracy runner, **not** a recreation of the common five-repeat paper timer.
For the separate cache comparison, see `benchmarks/cache/run_cache.py` and
the recorded `results/cache/` tables. The historical `mcwp_ablation/
benchmark_all_accuracy_runtime.py` covers earlier groups, so use the final
manifest for final 66-case claims.

## Training

`mcwp/train.py` is the short-horizon base trainer. The final fine-tuning
trainer is `mcwp_nccshift_v3/train_long100_tbptt.py` with the original
configuration recorded in `provenance/final_long100_training_record.json`:
initial base EMA100k, 100-step rollout, TBPTT chunk=20, 5,000 updates,
global batch=6, MSE-only auxiliary shift weight=0, initial LR=1e-5,
EMA decay=0.999. The paper chose EMA step 4,000.

Historical absolute machine paths in the training record are audit metadata,
not paths to use on a new machine. Use `--data-dir` for your regenerated HDF5
directory, `--init-checkpoint weights/mcwp_base_ema_100000.pt`, and an output
folder under `outputs/`. For a software-only HDF5 fixture, use
`mcwp/tests/make_fixture.py`. Synthetic fixture outputs do not validate paper
accuracy and must not be presented as FD trajectories.

## Recorded results

`results/final66_accuracy/updated_66_per_case.csv` has 66 cases × seven
models. `results/cache/` contains 66-case cache/FD timing and parity records.
`results/redsea_sensitivity/` contains two independent source result sets;
`summary_by_alpha.csv` has 8 conditions × four models. The supplementary
case, training-source, and source-preprocessing metadata are under
`results/supplement/`. Old three-seed and incomplete 51-case results were not
included as final-paper data.
