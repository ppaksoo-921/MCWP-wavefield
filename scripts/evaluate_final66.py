"""Re-evaluate the paper's 66 cases with their exact input-file checksums.

This is an accuracy runner; use the separate cache benchmark for timing.
Original wavefield case files are not stored in Git. See docs/DATA_SOURCES.md.
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "benchmarks/cache/paper_cases_final66.json"
SEEDS = ROOT / "results/supplement/evaluation_cases.csv"
GROUPS = ("ID", "Marmousi_GenWP", "Marmousi_source", "RedSea", "Kimberlina", "Pluto")
MODELS = ("base", "long100", "fno", "genwp_published", "genwp_retrained",
          "monolithic", "static_fusion")


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_path(item, args):
    if "repo_relative" in item:
        return args.data_root / item["repo_relative"]
    return args.marmousi_cases / item["marmousi_relative"]


def read_case(path):
    if path.suffix == ".mat":
        from scipy.io import loadmat
        d = loadmat(path)
        vp, ref = np.asarray(d["vp"], np.float32), np.asarray(d["snaps"], np.float32)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            vp = np.asarray(d["vp"], np.float32).copy()
            ref = np.asarray(d["snaps"], np.float32).copy()
    else:
        raise ValueError(f"Unsupported input: {path}")
    if ref.shape == (501, 128, 128):
        ref = ref[::5]
    if vp.shape != (128, 128) or ref.shape != (101, 128, 128):
        raise ValueError(f"Unexpected velocity or wavefield dimensions: {path}")
    if not np.isfinite(vp).all() or not np.isfinite(ref).all():
        raise ValueError(f"Non-finite input: {path}")
    return vp, ref


def rows_for_args(args):
    catalog = json.loads(CATALOG.read_text())["cases"]
    selected = [x for x in catalog if not args.groups or x["group"] in args.groups]
    if args.max_cases is not None:
        selected = selected[:args.max_cases]
    seeds = {r["case"]: int(r["genwp_noise_seed"])
             for r in csv.DictReader(SEEDS.open(newline=""))}
    if len(catalog) != 66 or len(selected) != len({x["case"] for x in selected}):
        raise RuntimeError("Case catalog has the wrong count or duplicate names")
    for item in selected:
        item["seed"] = seeds[item["case"]]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "GenWP-pub")
    parser.add_argument("--marmousi-cases", type=Path,
                        default=ROOT / "benchmarks/marmousi/results/marmousi_raw_direct_v1500_5000")
    parser.add_argument("--fno-source", type=Path,
                        default=ROOT / "third_party/neuraloperator_fno")
    parser.add_argument("--genwp-published", type=Path,
                        default=ROOT / "weights/genwp_published_trained_model.pt")
    parser.add_argument("--genwp-retrained", type=Path,
                        default=ROOT / "weights/genwp_retrained_ema_0.999_100000.pt")
    parser.add_argument("--groups", choices=GROUPS, action="append")
    parser.add_argument("--models", choices=MODELS, action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/final66")
    parser.add_argument("--list", action="store_true", help="List cases without importing PyTorch")
    parser.add_argument("--verify-inputs-only", action="store_true")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be at least one")
    cases = rows_for_args(args)
    if args.list:
        for item in cases:
            print(item["group"], item["case"], input_path(item, args))
        return

    missing = [str(input_path(x, args)) for x in cases if not input_path(x, args).is_file()]
    if missing:
        raise FileNotFoundError("Required case file(s) absent; see docs/DATA_SOURCES.md:\n" +
                                "\n".join(missing[:20]))
    for item in cases:
        path = input_path(item, args)
        actual = file_sha(path)
        if actual != item["expected_sha256"]:
            raise ValueError(f"Paper input checksum mismatch: {path}: {actual}")
    print(f"Verified {len(cases)} case-file checksums")
    if args.verify_inputs_only:
        return

    import torch
    sys.path.insert(0, str(ROOT / "benchmarks/marmousi"))
    from inference import load_model, rollout, metrics

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable")

    checkpoints = {
        "base": ROOT / "weights/mcwp_base_ema_100000.pt",
        "long100": ROOT / "weights/mcwp_final_ema_004000.pt",
        "fno": ROOT / "weights/fno_ema_100000.pt",
        "monolithic": ROOT / "weights/monolithic_ema_100000.pt",
        "static_fusion": ROOT / "weights/static_fusion_ema_100000.pt",
        "genwp_published": args.genwp_published,
        "genwp_retrained": args.genwp_retrained,
    }
    config = {
        "checkpoints": {k: str(v.resolve()) for k, v in checkpoints.items()},
        "genwp_repo": str(args.data_root.resolve()),
        "neuraloperator_source": str(args.fno_source.resolve()),
    }
    models = args.models or ["long100"]
    results = []
    for model_name in models:
        if model_name == "genwp_retrained":
            expected = "ceba73e1eaf379081cdbfad7f3f4a4e19696bb9cd276ad3a35474d22a6bfe6b1"
            if file_sha(args.genwp_retrained) != expected:
                raise ValueError("GenWP-Retrained checkpoint SHA256 differs from the paper")
        model, source = load_model(model_name, config, device)
        for item in cases:
            vp, reference = read_case(input_path(item, args))
            prediction = rollout(model, model_name, vp, reference, device,
                                 item["seed"], source.get("noise_channels", 1))
            score, _frames = metrics(prediction, reference)
            row = {"group": item["group"], "case": item["case"],
                   "model": model_name, "seed": item["seed"], **score}
            if model_name == "long100":
                row["mae_difference_from_record"] = (
                    score["mae"] - item["expected_mcwp_mae"])
            results.append(row)
            print(model_name, item["case"], f"MAE={score['mae']:.8f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in results for key in row)))
        writer.writeheader()
        writer.writerows(results)
    summary = []
    for model_name in models:
        by_group = {}
        for group in GROUPS:
            these = [x for x in results if x["model"] == model_name and x["group"] == group]
            if these:
                by_group[group] = statistics.mean(x["mae"] for x in these)
                summary.append({"model": model_name, "group": group,
                                "cases": len(these), "mae": by_group[group]})
        if len(by_group) == len(GROUPS):
            summary.append({"model": model_name, "group": "equal_group_aggregate",
                            "cases": 66, "mae": statistics.mean(by_group.values())})
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "group", "cases", "mae"])
        writer.writeheader()
        writer.writerows(summary)
    print("Saved", args.output_dir)


if __name__ == "__main__":
    main()
