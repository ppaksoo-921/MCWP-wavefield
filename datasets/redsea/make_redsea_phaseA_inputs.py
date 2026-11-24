from pathlib import Path
import numpy as np

ROOT = Path("dataset/redsea_ood_inputs")
RAW = ROOT / "raw"
OUT = ROOT / "phaseA"

OUT.mkdir(parents=True, exist_ok=True)

LOCS = np.array(
    [25, 38, 62, 105, 126],
    dtype=np.int32
)

for i in range(1, 5):

    src = RAW / f"redsea_{i:02d}.npy"

    vp = np.load(src).astype(np.float32)

    if vp.shape != (128, 128):
        raise RuntimeError(
            f"{src}: expected (128,128), got {vp.shape}"
        )

    if not np.isfinite(vp).all():
        raise RuntimeError(
            f"{src}: nonfinite velocity values"
        )

    dst = OUT / f"RedSea{i:02d}.npz"

    np.savez_compressed(
        dst,
        vp=vp,
        loc_x_list=LOCS,
    )

    print(
        dst,
        "| shape=", vp.shape,
        "| Vp=", float(vp.min()),
        "~", float(vp.max()),
        "| sources=", LOCS.tolist()
    )

print("\nDONE")
