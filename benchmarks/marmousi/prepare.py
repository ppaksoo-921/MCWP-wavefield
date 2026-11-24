"""Numerical helpers; preparation reads the full original model directly."""
from pathlib import Path
import numpy as np


def affine(v, source_low, source_high, target_low, target_high):
    if not (source_high > source_low and target_high > target_low):
        raise ValueError("Invalid affine-map endpoints.")
    a = np.asarray(v, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite velocity input.")
    return (target_low + (a - source_low) *
            ((target_high - target_low) / (source_high - source_low))).astype(np.float32)


def load_legacy(path):
    with np.load(path, allow_pickle=False) as d:
        if not {"vp", "snaps", "loc_x"}.issubset(d.files):
            raise ValueError("Expected an original Marmousi evaluation case: " + str(path))
        v = np.asarray(d["vp"], np.float32).copy()
        shape = d["snaps"].shape
        loc = int(np.asarray(d["loc_x"]).item())
    if v.shape != (128, 128) or shape not in ((101, 128, 128), (501, 128, 128)):
        raise ValueError("Unexpected original Marmousi case dimensions: " + str(path))
    if not np.isfinite(v).all():
        raise ValueError("Non-finite velocity: " + str(path))
    return v, loc


def prepare(c):
    from raw_original import prepare as direct_prepare
    return direct_prepare(c)
