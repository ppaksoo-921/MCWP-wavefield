from pathlib import Path
import argparse
import gc

import numpy as np
import torch

from fd_torch import acoustic_iso_torch


# ============================================================
# GenWP official simulation parameters
# ============================================================

DX = 10.0
DZ = 10.0
DT = 0.001
NT = 1001

FREQ = 15.0
N_PML = 50
TIME_DOWNSAMPLE = 10

NX = 128
NZ = 128
N_SHOTS = 5


def parse_name(path):
    family, model_id = path.stem.rsplit("_", 1)
    return family, int(model_id)


def expected_outputs(output_dir, family, model_id):
    return [
        output_dir / f"{family}_vel{model_id}_shot{shot_id}.npz"
        for shot_id in range(1, N_SHOTS + 1)
    ]


def validate_model_outputs(
    output_dir,
    family,
    model_id,
    vp_reference,
    loc_x_list,
):
    files = expected_outputs(
        output_dir,
        family,
        model_id,
    )

    for shot_id, path in enumerate(files, start=1):

        if not path.exists():
            raise RuntimeError(
                f"Missing output: {path}"
            )

        with np.load(path) as d:

            required = {"snaps", "vp", "loc_x"}

            if not required.issubset(d.files):
                raise RuntimeError(
                    f"Missing keys in {path}: {d.files}"
                )

            snaps = d["snaps"]
            vp = d["vp"]
            loc_x = int(d["loc_x"])

            if snaps.shape != (101, 128, 128):
                raise RuntimeError(
                    f"Bad snaps shape in {path}: "
                    f"{snaps.shape}"
                )

            if snaps.dtype != np.float32:
                raise RuntimeError(
                    f"Bad snaps dtype in {path}: "
                    f"{snaps.dtype}"
                )

            if vp.shape != (128, 128):
                raise RuntimeError(
                    f"Bad vp shape in {path}: "
                    f"{vp.shape}"
                )

            if not np.isfinite(snaps).all():
                raise RuntimeError(
                    f"NaN/Inf in {path}"
                )

            if np.max(np.abs(snaps)) == 0:
                raise RuntimeError(
                    f"All-zero wavefield in {path}"
                )

            expected_loc = int(
                loc_x_list[shot_id - 1]
            )

            if loc_x != expected_loc:
                raise RuntimeError(
                    f"Source mismatch in {path}: "
                    f"{loc_x} != {expected_loc}"
                )

            if not np.allclose(
                vp,
                vp_reference,
                rtol=0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"Velocity mismatch in {path}"
                )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="../phaseA_inputs/velocity_models",
    )

    parser.add_argument(
        "--output-dir",
        default="../phaseA_outputs/raw_npz",
    )

    parser.add_argument(
        "--family",
        default=None,
        help="Optional: process one family only",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: number of velocity models to process",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of parallel GPU workers",
    )

    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This worker's shard ID",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available."
        )

    device = "cuda"

    print("========================================")
    print("Phase A GenWP wavefield generation")
    print("========================================")
    print("Input :", input_dir.resolve())
    print("Output:", output_dir.resolve())
    print("Torch :", torch.__version__)
    print("GPU   :", torch.cuda.get_device_name(0))

    files = list(
        input_dir.glob("*.npz")
    )

    files = sorted(
        files,
        key=lambda p: parse_name(p)
    )

    if args.family is not None:
        files = [
            p for p in files
            if parse_name(p)[0] == args.family
        ]

    # Multi-GPU sharding
    files = files[
        args.shard_id::args.num_shards
    ]

    if args.limit is not None:
        files = files[:args.limit]

    print("Models assigned:", len(files))
    print("Shard:", args.shard_id, "/", args.num_shards)
    print("========================================")

    completed = 0
    skipped = 0

    for count, input_path in enumerate(
        files,
        start=1,
    ):

        family, model_id = parse_name(
            input_path
        )

        expected = expected_outputs(
            output_dir,
            family,
            model_id,
        )

        # ----------------------------------------
        # Resume support
        # ----------------------------------------

        if all(p.exists() for p in expected):

            try:
                with np.load(input_path) as d:
                    vp = d["vp"].astype(
                        np.float32
                    )
                    loc_x_list = (
                        d["loc_x_list"]
                        .astype(int)
                        .tolist()
                    )

                validate_model_outputs(
                    output_dir,
                    family,
                    model_id,
                    vp,
                    loc_x_list,
                )

                skipped += 1

                print(
                    f"[{count}/{len(files)}] "
                    f"SKIP {family}_{model_id}"
                )

                continue

            except Exception as e:

                print(
                    f"[{count}/{len(files)}] "
                    f"Existing output failed QC; "
                    f"rerunning {family}_{model_id}"
                )

                print("Reason:", e)

        # ----------------------------------------
        # Load prepared Phase-A input
        # ----------------------------------------

        with np.load(input_path) as d:

            vp = d["vp"].astype(
                np.float32
            )

            loc_x_list = (
                d["loc_x_list"]
                .astype(int)
                .tolist()
            )

        if vp.shape != (128, 128):
            raise RuntimeError(
                f"Bad velocity shape: "
                f"{input_path} {vp.shape}"
            )

        if len(loc_x_list) != 5:
            raise RuntimeError(
                f"Expected 5 sources: "
                f"{input_path} "
                f"{loc_x_list}"
            )

        if not np.isfinite(vp).all():
            raise RuntimeError(
                f"NaN/Inf velocity: "
                f"{input_path}"
            )

        print()
        print(
            f"[{count}/{len(files)}] "
            f"{family}_{model_id}"
        )

        print(
            "Source locations:",
            loc_x_list
        )

        print(
            "Velocity range:",
            float(vp.min()),
            float(vp.max())
        )

        # Constant-density acoustic model
        rho = np.ones(
            (NZ, NX),
            dtype=np.float32
        )

        # ----------------------------------------
        # Official GenWP FD solver
        # ----------------------------------------

        solver = acoustic_iso_torch(
            dx=DX,
            dz=DZ,
            dt=DT,
            nt=NT,
            loc_x_list=loc_x_list,
            freq=FREQ,
            n_pml=N_PML,
            vel=vp,
            rho=rho,
            device=device,
        )

        solver.set_method("FD10")

        # IMPORTANT:
        # run_fd adds +1 when naming data_id.
        # Therefore model_id=2 -> data_id=1.
        solver.run_fd(
            data_id=model_id - 1,
            data_folder=str(output_dir),
            md=family,
            time_downsample=TIME_DOWNSAMPLE,
        )

        # ----------------------------------------
        # QC immediately after simulation
        # ----------------------------------------

        validate_model_outputs(
            output_dir,
            family,
            model_id,
            vp,
            loc_x_list,
        )

        completed += 1

        print(
            f"QC PASS: {family}_{model_id}"
        )

        del solver
        gc.collect()

        if count % 50 == 0:
            torch.cuda.empty_cache()

    print()
    print("========================================")
    print("GENERATION FINISHED")
    print("========================================")
    print("Newly simulated models:", completed)
    print("Skipped existing models:", skipped)
    print("Total assigned models :", len(files))


if __name__ == "__main__":
    main()
