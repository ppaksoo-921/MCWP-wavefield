"""Reconstruct the exact four Red Sea evaluation velocity grids.

The distributed inputs are already 128 x 128. Their earlier extraction from the
2801 x 9441 regional source is not represented by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = {
    1: "a0d7763c0e4c941500f5ff9d2dc36581162f5d7e323901070838e2f98c5d3e14",
    2: "8d251aaa999409e439bdb11fcdf804474ea00d090fe9518cab2f66c9e75032e1",
    3: "fe898ae777ecd0694e87856d41578efb76f0a3548c0dacec9c6be1c99c7837ed",
    4: "eec21f50b957b75aca4f01a0e7f5871156f27eb57b736f44ef7442dd4420ad03",
}
EVALUATION_SHA256 = {
    1: "670ce2ca32408c1bf1d3f44e04d5ad835d48fa2f030f3aa427bd9e62d4ba788b",
    2: "4a01cb1e081fec3087ed2b5f427f67acd60d79fa1d91520bc4ef49dd8a42ece6",
    3: "fd6934ede36636188d28d4ed8b74049c4e82a5e72ec52aac71510846ac15cd60",
    4: "55741c320293e11ca918aacc9edebc79926b7bbc08f4846b2d279bf7e10b83d5",
}


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def prepare(raw_dir=HERE / "raw", mapping_file=HERE / "range_mapping.json",
            output_dir=None):
    mapping = json.loads(Path(mapping_file).read_text())["mapping"]
    bounds = (mapping["old_global_min"], mapping["old_global_max"],
              mapping["new_global_min"], mapping["new_global_max"])
    if tuple(float(x) for x in bounds) != (1400.0, 5300.0, 1450.0, 5000.0):
        raise ValueError("The supplied range mapping differs from the paper.")
    if abs(mapping["scale"] - 3550.0 / 3900.0) > 1e-12:
        raise ValueError("Unexpected mapping scale.")

    outputs = {}
    for model_id in range(1, 5):
        source = Path(raw_dir) / f"redsea_{model_id:02d}.npy"
        vp = np.load(source, allow_pickle=False)
        if vp.shape != (128, 128) or vp.dtype != np.float32:
            raise ValueError(f"Unexpected shape/dtype: {source}: {vp.shape}, {vp.dtype}")
        if not np.isfinite(vp).all() or array_sha(vp) != SOURCE_SHA256[model_id]:
            raise ValueError(f"Source velocity checksum mismatch: {source}")

        # The single-precision operation order matches the evaluated grids.
        matched = (np.float32(1450.0) +
                   (vp - np.float32(1400.0)) * np.float32(3550.0 / 3900.0))
        if array_sha(matched) != EVALUATION_SHA256[model_id]:
            raise ValueError(f"Prepared velocity checksum mismatch: model {model_id}")
        outputs[model_id] = matched
        if output_dir is not None:
            dest = Path(output_dir)
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / f"RedSea_vel{model_id}.npy", matched,
                    allow_pickle=False)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=HERE / "raw")
    parser.add_argument("--mapping", type=Path, default=HERE / "range_mapping.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "velocity_models")
    args = parser.parse_args()
    for model_id, values in prepare(args.raw_dir, args.mapping, args.out_dir).items():
        print(f"RedSea_vel{model_id}: {values.min():.1f}..{values.max():.1f} m/s, "
              f"sha256={array_sha(values)}")
