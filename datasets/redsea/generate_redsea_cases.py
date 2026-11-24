"""Run the original FD10 solver for 20 Red Sea benchmark cases.

Use an empty output directory; existing reference wavefields are never overwritten.
The final case-file hashes can depend on the original CUDA/PyTorch environment.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from fd_torch import acoustic_iso_torch  # noqa: E402

from prepare_redsea import prepare  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    if not args.models or any(i not in range(1, 5) for i in args.models):
        parser.error("--models must list model identifiers 1 through 4")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.out_dir / f"RedSea_vel{i}_shot{s}.npz"
                for i in args.models for s in range(1, 6)
                if (args.out_dir / f"RedSea_vel{i}_shot{s}.npz").exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing reference: {existing[0]}")
    velocity = prepare()
    for model_id in args.models:
        solver = acoustic_iso_torch(
            dx=10.0, dz=10.0, dt=0.001, nt=1001,
            loc_x_list=[25, 38, 62, 105, 126], freq=15.0, n_pml=50,
            vel=velocity[model_id], rho=np.ones((128, 128), np.float32),
            device=args.device,
        )
        solver.set_method("FD10")
        solver.run_fd(data_id=model_id - 1, data_folder=str(args.out_dir),
                      md="RedSea", time_downsample=10)
    for model_id in args.models:
        for shot, expected_x in enumerate([25, 38, 62, 105, 126], 1):
            path = args.out_dir / f"RedSea_vel{model_id}_shot{shot}.npz"
            with np.load(path, allow_pickle=False) as z:
                if z["vp"].shape != (128, 128) or z["snaps"].shape != (101, 128, 128):
                    raise ValueError(f"Unexpected case dimensions: {path}")
                if int(np.asarray(z["loc_x"]).item()) != expected_x:
                    raise ValueError(f"Unexpected source coordinate: {path}")
            print(path)


if __name__ == "__main__":
    main()
