from pathlib import Path

import numpy as np
import torch

from fd_torch import acoustic_iso_torch


DATASET_ROOT = Path.home() / "Wage_surrogate/GenWP-pub/dataset"

ROOT = DATASET_ROOT / "kimberlina_ood_year0"

VP_DIR = ROOT / "velocity_models_128"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

dx = 10.0
dz = 10.0

dt = 0.001
nt = 1001

freq = 15.0
n_pml = 50

time_downsample = 10

loc_x_list = [
    25,
    38,
    62,
    105,
    126,
]

rho = np.ones(
    (128, 128),
    dtype=np.float32,
)


print(f"Device: {device}")
print(f"Source locations: {loc_x_list}")
print()


for vel_id in [1, 2]:

    vp_path = VP_DIR / f"Kimberlina_{vel_id}.npz"

    if not vp_path.exists():
        raise FileNotFoundError(vp_path)

    with np.load(vp_path) as data:
        vp = data["vp"].astype(np.float32)

    if vp.shape != (128, 128):
        raise RuntimeError(
            f"Unexpected velocity shape for vel{vel_id}: {vp.shape}"
        )

    if not np.all(np.isfinite(vp)):
        raise RuntimeError(
            f"NaN or Inf detected in vel{vel_id}"
        )

    print("=" * 70)
    print(f"Kimberlina velocity model {vel_id}")
    print(f"Shape : {vp.shape}")
    print(f"Min   : {vp.min():.3f} m/s")
    print(f"Max   : {vp.max():.3f} m/s")
    print("=" * 70)

    solver = acoustic_iso_torch(
        dx=dx,
        dz=dz,
        dt=dt,
        nt=nt,
        loc_x_list=loc_x_list,
        freq=freq,
        n_pml=n_pml,
        vel=vp,
        rho=rho,
        device=device,
    )

    solver.set_method("FD10")

    solver.run_fd(
        data_id=vel_id - 1,
        data_folder=str(OUTPUT_DIR),
        md="Kimberlina",
        time_downsample=time_downsample,
    )


print()
print("FD generation completed.")
print(f"Output directory: {OUTPUT_DIR}")


output_files = sorted(
    OUTPUT_DIR.glob("Kimberlina_vel*_shot*.npz")
)

print(f"Generated files: {len(output_files)}")

if len(output_files) != 10:
    raise RuntimeError(
        f"Expected 10 output files, found {len(output_files)}"
    )


expected_locations = {
    shot_id + 1: loc_x
    for shot_id, loc_x in enumerate(loc_x_list)
}


for path in output_files:

    with np.load(path) as data:
        snaps = data["snaps"]
        vp_out = data["vp"]
        loc_x = int(np.asarray(data["loc_x"]).item())

    if snaps.shape != (101, 128, 128):
        raise RuntimeError(
            f"{path.name}: bad snaps shape {snaps.shape}"
        )

    if vp_out.shape != (128, 128):
        raise RuntimeError(
            f"{path.name}: bad vp shape {vp_out.shape}"
        )

    if not np.all(np.isfinite(snaps)):
        raise RuntimeError(
            f"{path.name}: NaN or Inf in snapshots"
        )

    shot_id = int(
        path.stem.split("shot")[-1]
    )

    expected_loc = expected_locations[shot_id]

    if loc_x != expected_loc:
        raise RuntimeError(
            f"{path.name}: loc_x={loc_x}, expected={expected_loc}"
        )

    print(
        f"{path.name:30s} "
        f"snaps={snaps.shape} "
        f"loc_x={loc_x:3d} "
        f"peak={np.abs(snaps).max():.6e}"
    )


print()
print("All 10 files passed validation.")
