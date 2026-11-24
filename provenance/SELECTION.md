# Selection from the study archives

Included:

- Exact original MCWP model/base training source from `mcwp_release(1).zip`.
- Final long-horizon trainer and its five hashed implementation components
  from `mcwp_nccshift_v3.zip`, with original configuration and base/final EMA
  checkpoints (`config(1).json`, `ema_100000.pt`, `ema_004000.pt`).
- Structural-control model source from `mcwp_ablation_release(1).zip` and
  comparison checkpoints for monolithic/static fusion from `number1.zip`.
- FNO wrapper and training code from `fno_baseline.zip`, with the FNO weight
  from `number1.zip`. NeuralOperator remains an external pinned dependency.
- Core FD solver, training data conversion, and Kimberlina builders from
  `1124_1450.zip`. Pluto source/manifest/prepared 128-grid velocities from
  its nested `Pluto_OOD_mps_Package.zip`, excluding the full converted SEG-Y.
- Four matching Red Sea 128-grid sources, range mapping, and exact
  reconstruction script. The velocity hashes were crosschecked against all
  20 primary Red Sea cases.
- Final 66-case manifest, selected per-case/summary results, cache timing,
  and Red Sea range-sensitivity tables. Table S5's 96 displayed metrics
  matched the report CSV at six decimal places.
- Three supplementary metadata files: `evaluation_cases.csv`,
  `training_sources.csv`, and `source_preprocessing.json`.

Excluded:

- Experimental NCC-shift branches and late three-seed reports that were not
  used for the final paper claims, local run folders, machine-specific logs,
  compiled Python caches, screenshots, figure exports, and paper PDF/LaTeX.
- Full Pluto converted SEG-Y/NPY, raw pressure trajectories, original training
  HDF5 collection, and copied GenWP/NeuralOperator source trees.
- GenWP-Retrained 503 MB checkpoint from `number2.zip`; publish it as a GitHub
  Release asset after verifying its SHA256. The upstream published GenWP
  checkpoint is referenced, not copied.

The historical checkpoint identifiers and original experiment configuration
remain in JSON for audit. `SHA256SUMS.txt` hashes the selected distribution
files after any new documentation or wrapper scripts were added. The local
environment did not have PyTorch/CUDA; complete model/FD inference must be
checked on the author's original research server.
