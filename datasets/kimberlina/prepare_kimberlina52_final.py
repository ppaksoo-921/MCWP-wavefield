from pathlib import Path
import csv
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


DATASET_ROOT = Path.home() / "Wage_surrogate/GenWP-pub/dataset"
ROOT = DATASET_ROOT / "kimberlina_ood_year0"

ZIP_PATH = ROOT / "raw/vp_year0.zip"

OUT_DIR = ROOT / "velocity_models_128"
PREVIEW_DIR = ROOT / "final_previews"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

NZ = 350
NX = 600

TARGET_NZ = 128
TARGET_NX = 128

SLIDE = 52


with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    matches = [
        name for name in zf.namelist()
        if name.endswith(f"vp_year0_slide{SLIDE}.bin")
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one slide {SLIDE} file, found {matches}"
        )

    name = matches[0]
    raw = zf.read(name)


data = np.frombuffer(raw, dtype="<f4")

if data.size != NZ * NX:
    raise RuntimeError(
        f"Unexpected number of values: {data.size}, "
        f"expected {NZ * NX}"
    )

vp = data.reshape((NZ, NX), order="F").astype(np.float32)

if not np.all(np.isfinite(vp)):
    raise RuntimeError("NaN or Inf detected in velocity model")

print(f"Source file : {name}")
print(f"Raw shape   : {vp.shape}")
print(f"Raw Vp min  : {vp.min():.3f} m/s")
print(f"Raw Vp max  : {vp.max():.3f} m/s")
print(f"Raw Vp mean : {vp.mean():.3f} m/s")


crops = [
    {
        "vel_id": 1,
        "name": "left",
        "x0": 0,
        "x1": 350,
    },
    {
        "vel_id": 2,
        "name": "right",
        "x0": 250,
        "x1": 600,
    },
]

manifest_rows = []


for crop_info in crops:
    vel_id = crop_info["vel_id"]
    crop_name = crop_info["name"]
    x0 = crop_info["x0"]
    x1 = crop_info["x1"]

    crop = vp[:, x0:x1]

    if crop.shape != (350, 350):
        raise RuntimeError(
            f"{crop_name}: unexpected crop shape {crop.shape}"
        )

    scale_z = TARGET_NZ / crop.shape[0]
    scale_x = TARGET_NX / crop.shape[1]

    vp128 = zoom(
        crop,
        zoom=(scale_z, scale_x),
        order=1,
    ).astype(np.float32)

    if vp128.shape != (TARGET_NZ, TARGET_NX):
        raise RuntimeError(
            f"{crop_name}: unexpected resized shape {vp128.shape}"
        )

    out_npz = OUT_DIR / f"Kimberlina_{vel_id}.npz"

    np.savez_compressed(
        out_npz,
        vp=vp128,
    )

    print()
    print(f"[{crop_name.upper()}]")
    print(f"Original crop : z=0:350, x={x0}:{x1}")
    print(f"Crop shape    : {crop.shape}")
    print(f"Resized shape : {vp128.shape}")
    print(f"Vp min        : {vp128.min():.3f} m/s")
    print(f"Vp max        : {vp128.max():.3f} m/s")
    print(f"Saved         : {out_npz}")

    manifest_rows.append(
        [
            vel_id,
            crop_name,
            SLIDE,
            0,
            350,
            x0,
            x1,
            crop.shape[0],
            crop.shape[1],
            vp128.shape[0],
            vp128.shape[1],
            float(vp128.min()),
            float(vp128.max()),
            float(vp128.mean()),
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        vp128,
        interpolation="nearest",
        aspect="equal",
        vmin=float(vp.min()),
        vmax=float(vp.max()),
    )

    ax.set_title(
        f"Kimberlina Slide 52 - {crop_name.capitalize()}\n"
        f"128 x 128"
    )
    ax.set_xlabel("X index")
    ax.set_ylabel("Z index")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P-wave velocity (m/s)")

    fig.tight_layout()

    fig.savefig(
        PREVIEW_DIR / f"Kimberlina_{vel_id}_{crop_name}_128.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


manifest_path = ROOT / "kimberlina52_manifest.csv"

with open(manifest_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "vel_id",
            "crop",
            "source_slide",
            "z_start",
            "z_end",
            "x_start",
            "x_end",
            "original_nz",
            "original_nx",
            "final_nz",
            "final_nx",
            "vp_min_mps",
            "vp_max_mps",
            "vp_mean_mps",
        ]
    )

    writer.writerows(manifest_rows)


fig, ax = plt.subplots(figsize=(12, 7))

im = ax.imshow(
    vp,
    interpolation="nearest",
    aspect="equal",
)

ax.axvline(350, linewidth=2, linestyle="--")
ax.axvline(250, linewidth=2, linestyle="--")

ax.set_title("Kimberlina Year-0 Slide 52")
ax.set_xlabel("X index")
ax.set_ylabel("Z index")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("P-wave velocity (m/s)")

fig.tight_layout()

fig.savefig(
    PREVIEW_DIR / "Kimberlina_slide52_full.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Preparation completed.")
print(f"Manifest: {manifest_path}")
