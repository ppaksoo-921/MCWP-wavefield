# Data sources and reproducibility boundary

## Published case inputs

`benchmarks/cache/paper_cases_final66.json` records the paths, SHA256 hashes,
group labels, and recorded MCWP MAE for all 66 final cases. The companion
`results/supplement/evaluation_cases.csv` records source coordinates and GenWP
noise seeds. Do not replace the final Marmousi direct-source cases with an
earlier 15-case or three-seed dataset of the same size.

| Group | Cases | Input path under the external `GenWP-pub/` tree | Origin |
| --- | ---: | --- | --- |
| ID | 4 | `dataset/test/{Overthrust,SEGEAGE}/shot*.mat` | Official GenWP test files. |
| Marmousi (GenWP) | 2 | `dataset/test/Marmousi/shot*.mat` | Official GenWP test files. |
| Marmousi (this study) | 15 | `benchmarks/marmousi/results/marmousi_raw_direct_v1500_5000/fd_cases/*.npz` | `benchmarks/marmousi/` code and original Marmousi source binary (not redistributed here). |
| Red Sea | 20 | `dataset/redsea_ood_range_matched/outputs/*.npz` | Four supplied 128×128 arrays, global range mapping, and FD generator. |
| Kimberlina | 10 | `dataset/kimberlina_ood_outputs/*.npz` | Kimberlina year-0 slide-52 binary, preparation and FD generator. |
| Pluto | 15 | `dataset/pluto_ood_outputs/*.npz` | Pluto 1.5 original SEG-Y, preparation and FD generator. |

The user must supply or regenerate the reference wavefield files at these
paths. Their large pressure arrays were not committed to Git. On a different
PyTorch/CUDA environment, FD file checksums can differ even with matching
physical settings. In that event compare the numerical fields and disclose
the environment; do not silently claim exact original-case reproduction.

## Red Sea velocity inputs

`datasets/redsea/raw/redsea_01.npy` to `redsea_04.npy` are the four prepared
128×128 FP32 arrays spanning 1400–5300 m/s. The command below applies one
global affine mapping in FP32:

```bash
python datasets/redsea/prepare_redsea.py
```

The mapping is `v' = 1450 + (v - 1400) * (3550 / 3900)` m/s. All four output
array checksums match the original 20-case Red Sea experiment's velocity
checksums. Source positions are x=25,38,62,105,126, at z=1, with a common
128×128 FD grid; call `datasets/redsea/generate_redsea_cases.py` for the FD
references. The upstream extraction from a 2801×9441 source to the four
provided 128×128 arrays is **described in the article's supplement but was
not independently reconstructed or verified from its original source file**.
This repository does not claim end-to-end reconstruction of that upstream
stage. The supplied prepared arrays let reviewers evaluate the paper's actual
20 cases once corresponding FD references are regenerated.

## Other geological inputs

- The Marmousi raw source, crop coordinates, and global mapping are recorded in
  `results/supplement/source_preprocessing.json`, and the direct-source
  evaluation scripts are in `benchmarks/marmousi/`. The source binary itself
  is not included.
- `datasets/kimberlina/prepare_kimberlina52_final.py` reads
  `kimberlina_ood_year0/raw/vp_year0.zip` and crops two slide-52 regions.
  `generate_kimberlina52_fd.py` creates ten FD cases. The original archive is
  available from NETL's Kimberlina 1.2 dataset, DOI `10.18141/1832899`.
- `datasets/pluto/prepare_pluto.py` converts the original Pluto 1.5 SEG-Y
  velocity from ft/s to m/s and prepares three 128×128 regions. The manifest
  records the original source checksum and crop extents; the three prepared
  grids are supplied, with SHA256 values matching the final Pluto report.
  The converted full SEG-Y/NPY was deliberately excluded from Git.

## Training data

The MCWP training manifest lists 7,871 velocity patches, each sampled at five
source locations (39,355 trajectories). The complete HDF5 collection is not
included. Use the official GenWP source velocity dataset and the scripts in
`datasets/training/` for FD reference generation and NPZ-to-HDF5 conversion.
`results/supplement/training_sources.csv` lists the trajectory identities.
The upstream construction of all original geological patches is delegated to
the original velocity-dataset publication; this repo does not assert that
the patches can be independently reconstructed from just this Git tree.

## Attribution and binary redistribution

The original Red Sea, Marmousi, Kimberlina, Pluto, and GenWP datasets have
their own attribution and access terms. Before publication, the repository
owner should check permission to redistribute prepared derivative velocity
arrays. If their terms prohibit redistribution, remove the affected binary
arrays, supply authorized download/reconstruction instructions and synthetic
software-test input, and explicitly explain the resulting limitation in the
manuscript and repository. The MCWP software license does not grant rights to
unrelated source datasets.
