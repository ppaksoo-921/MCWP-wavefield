# MCWP: medium-conditioned autoregressive acoustic wavefields

Source code, trained MCWP weights, prepared example media, and numerical audit
tables for the article *MCWP: Medium-Conditioned Autoregressive Prediction of
Acoustic Wavefields*. The evaluated model predicts 100 pressure frames from a
fixed velocity grid and the initial pressure field. Source and third-party
data are credited in [Data sources](docs/DATA_SOURCES.md).

## Contents

| Directory | Contents |
| --- | --- |
| `mcwp/` | Original MCWP model and short-horizon trainer. |
| `mcwp_nccshift_v3/` | Final long-horizon TBPTT trainer used with MSE-only configuration. |
| `mcwp_ablation/` | Structural-control architectures, training, and legacy benchmark code. |
| `fno_baseline/` | Task wrapper and trainer; NeuralOperator itself is an external dependency. |
| `datasets/` | FD10 solver, data conversion, four prepared Red Sea input grids, Kimberlina and Pluto builders. |
| `benchmarks/` | Final 66-case manifest, MCWP/FD cache benchmark, and Marmousi evaluation adapters. |
| `scripts/` | A strict SHA-checked 66-case accuracy runner. |
| `results/` | Selected CSV/JSON audit records; no raw wavefield trajectories. |
| `weights/` | Five checkpoints each below GitHub's regular-file limit; large GenWP weight is distributed separately. |

## Quick check without a GPU

Use Python 3.10 and NumPy 1.26.4. From this repository root:

```bash
python -m pip install numpy==1.26.4
python -m unittest discover -s tests -v
python datasets/redsea/prepare_redsea.py
python scripts/evaluate_final66.py --groups RedSea --max-cases 2 --list
```

The Red Sea preparation verifies byte-level checksums of the four supplied
128×128 inputs and the four evaluated 1450–5000 m/s velocity grids. The last
command only lists study cases; it does not run the network or FD solver.

## Environment and model evaluation

The reported GPU experiments used Linux, Python 3.10.16, PyTorch 1.12.1,
NumPy 1.26.4, FP32/TF32 disabled, and an NVIDIA RTX 3090. Install dependencies
compatible with your CUDA driver and follow [User guide](docs/USER_GUIDE.md).
The original GenWP implementation is an external dependency from
[DeepWave-KAUST/GenWP-pub](https://github.com/DeepWave-KAUST/GenWP-pub).
The FNO implementation must be checked out at NeuralOperator commit
`8a0f5260983efffbe6e8ec39c2d095dba8cb2505`.

Create a software-test fixture (not physical FD reference data):

```bash
python mcwp/tests/make_fixture.py outputs/synthetic_h5 --files 2
python mcwp/tests/test_core.py
```

With the source model, matching case files, and GPU environment installed,
generate the Red Sea FD references into the GenWP dataset tree:

```bash
python datasets/redsea/generate_redsea_cases.py \
  --out-dir GenWP-pub/dataset/redsea_ood_range_matched/outputs --device cuda:0
python scripts/evaluate_final66.py --data-root GenWP-pub \
  --groups RedSea --models long100 --device cuda:0
```

The 66-case runner requires every selected input file to match the published
case-file SHA256. Original FD outputs can vary with changes to the environment;
do not bypass a mismatch and describe such a run as the original paper result.
It reports accuracy, not the paper's separately timed runtime benchmark.

## Data and limits

The four distributed Red Sea velocity arrays are **prepared 128×128 inputs**;
their extraction from the original 2801×9441 regional array has not been
independently recovered. The original complete training HDF5 collection and
all 66 pressure-reference files are too large for this Git tree. The article's
original case checksums and a small synthetic software test fixture are
provided. The raw data origin, regeneration routes, and limits are documented
in [Data sources](docs/DATA_SOURCES.md). The recorded result tables can be
inspected with no GPU.

The 503 MB GenWP-Retrained EMA checkpoint exceeds GitHub's normal Git file
limit. Obtain it from this repository's **Releases** page as the
`genwp_retrained_ema_0.999_100000.pt` asset after the repository owner
publishes release `v1.0.0`. The GenWP-Published checkpoint comes from the
upstream project. All checkpoint SHA256 values appear in
[`weights/manifest.json`](weights/manifest.json).

See [User guide](docs/USER_GUIDE.md) for inputs, outputs, and flags, and
[Repository upload guide (Korean)](docs/GIT_UPLOAD_KO.md) for publication steps.

## License and citation

This study's code is provided under the MIT license in `LICENSE` (copyright
holder and authorization must be reviewed by the authors before publication).
The external GenWP code and the FD solver retain their upstream MIT notice;
other velocity datasets retain their own source terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Cite the article and source
datasets when reusing this material.
