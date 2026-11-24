"""CPU-only configuration, provenance and output helpers."""
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
REGIONS = ("Left", "Middle", "Right")
SHOTS = (25, 38, 62, 105, 126)
GROUP = "marmousi_original_v1500_5000"
DISPLAY = {
    "base": "MCWP-Base", "long100": "MCWP-Long100", "fno": "FNO",
    "genwp_published": "GenWP-Published", "genwp_retrained": "GenWP-Retrained",
    "monolithic": "Monolithic", "static_fusion": "Static fusion",
}
FD = dict(dx_m=10.0, dz_m=10.0, dt_s=0.001, nt=1001, frequency_hz=15.0,
          pml_cells=50, snapshot_stride=10, density=1.0, source_z_index=1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a, dtype="<f4")
    return hashlib.sha256(a.tobytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def confined(path):
    p = Path(path).expanduser().resolve()
    if not p.is_relative_to(HERE):
        raise ValueError("New outputs must stay inside the installed package: " + str(HERE))
    return p


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jwrite(path, value):
    p = confined(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.%d" % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty result table: " + str(path))
    p = confined(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_npz(path, **arrays):
    import numpy as np
    p = confined(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.%d" % os.getpid())
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)


def config(path=None):
    p = Path(path).expanduser().resolve() if path else HERE / "config.json"
    c = jread(p)
    root = Path(c["study_root"]).expanduser().resolve()
    c["study_root"] = str(root)
    for key in ("metadata", "original_binary", "legacy_cases_dir", "hard_models_dir",
                "genwp_repo", "neuraloperator_source"):
        q = Path(c[key]).expanduser()
        c[key] = str((root / q).resolve() if not q.is_absolute() else q.resolve())
    c["checkpoints"] = {k: str((root / Path(v)).resolve()) for k, v in c["checkpoints"].items()}
    work = Path(c["work_dir"])
    c["work_dir"] = str(confined(HERE / work if not work.is_absolute() else work))
    if (c["target_velocity_min_mps"], c["target_velocity_max_mps"]) != (1500.0, 5000.0):
        raise ValueError("This release preserves the 1500 m/s minimum and maps the maximum to 5000 m/s.")
    if not c["models"] or len(set(c["models"])) != len(c["models"]):
        raise ValueError("Provide unique model keys in config.json.")
    if any(k not in DISPLAY or k not in c["checkpoints"] for k in c["models"]):
        raise ValueError("Unknown or unconfigured model.")
    public = (root / "GenWP-pub/dataset/test/Marmousi").resolve()
    legacy = Path(c["legacy_cases_dir"])
    if legacy == public or legacy.is_relative_to(public):
        raise ValueError("GenWP public Marmousi is not the separately obtained original Marmousi.")
    return c


def bootstrap():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable, name in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"),
                           ("TORCH_HOME", "torch"), ("CUDA_CACHE_PATH", "cuda"),
                           ("XDG_CACHE_HOME", "xdg")):
        p = HERE / "_cache" / name
        p.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(p)
    tempfile.tempdir = os.environ["TMPDIR"]


def vendor_hashes():
    return {p.name: sha(p) for p in sorted(VENDOR.glob("*.py"))}


def implementation_hashes():
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}


def public_snapshot(c):
    root = Path(c["study_root"]) / "GenWP-pub/dataset/test/Marmousi"
    return {str(root / ("shot%d.mat" % s)): sha(root / ("shot%d.mat" % s))
            for s in (1, 2) if (root / ("shot%d.mat" % s)).is_file()}


def verify_public(before):
    for p, expected in before.items():
        if not Path(p).is_file() or sha(p) != expected:
            raise RuntimeError("GenWP PUBLIC Marmousi changed during the run: " + p)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_torch(path):
    import torch
    kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False
    return torch.load(path, **kw)


def select_cases(manifest, smoke=False):
    return [x for x in manifest["cases"] if not smoke or x["shot"] == 1]


def load_prepared(c):
    m = jread(Path(c["work_dir"]) / "preparation.json")
    if m["vendor_hashes"] != vendor_hashes() or m["implementation_hashes"] != implementation_hashes():
        raise RuntimeError("Implementation changed after preparation. Use a new work_dir and prepare again.")
    if m["legacy_cases_dir"] != c["legacy_cases_dir"]:
        raise RuntimeError("Source directory differs from the prepared experiment.")
    if m["raw_metadata_sha256"] != sha(c["metadata"]):
        raise RuntimeError("Original-model metadata changed after preparation.")
    if m.get("source_mode") == "direct_full_original_file":
        src = m["raw_binary_audit"]
        if c["original_binary"] != src["path"] or sha(c["original_binary"]) != src["file_sha256"]:
            raise RuntimeError("Full original input file changed after preparation.")
        if Path(c["original_binary"]).suffix.lower() == ".bin" and c.get("raw_binary_layout", "xz_C") != src["layout"]:
            raise RuntimeError("Original binary layout changed after preparation.")
    verify_public(m["protected_genwp_public_files"])
    for item in m["models"]:
        p = Path(c["work_dir"]) / item["file"]
        if sha(p) != item["file_sha256"]:
            raise RuntimeError("Prepared velocity file changed: " + str(p))
    for item in m["cases"]:
        if sha(item["legacy_file"]) != item["legacy_file_sha256"]:
            raise RuntimeError("An original evaluation case changed: " + item["legacy_file"])
    return m
