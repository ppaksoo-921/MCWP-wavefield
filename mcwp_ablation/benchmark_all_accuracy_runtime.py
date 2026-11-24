#!/usr/bin/env python3
"""
Unified benchmark for learned wave propagators and FD10.

Primary study outputs:
- learned-model prediction error vs stored GT
- learned-model runtime vs FD10 runtime
- per-frame learned-model error

Default datasets:
1) ID: Overthrust shot1/2 + SEGEAGE shot1/2
2) Official Marmousi OOD: shot1/2
3) New Marmousi OOD: all MarmousiMain*.npz
4) Range-matched Red Sea OOD: all RedSea_vel*_shot*.npz

FD recomputation-vs-stored-GT is only an optional sanity check:
  --fd-sanity none   : do not compute it (default)
  --fd-sanity first  : first case of each dataset only
  --fd-sanity all    : every case

Runtime protocol:
- one CUDA GPU
- checkpoint loading excluded
- disk I/O excluded
- CUDA synchronization before/after each timed call
- "cold" single-case latency: medium-dependent preprocessing included
- FD stores only the 101 required output snapshots
"""

import argparse
import csv
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch

from common import trusted_load
from reference_model import ModelConfig, MediumConditionedWavePropagator
from model import build_model


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=str(Path.home() / "Wage_surrogate/GenWP-pub"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dataset",
        action="append",
        choices=["id", "marmousi_official", "marmousi_new", "redsea"],
        default=[],
        help="Repeatable. If omitted, all four groups are used.",
    )
    p.add_argument(
        "--mcwp", action="append", nargs=2, metavar=("LABEL", "CHECKPOINT"), default=[]
    )
    p.add_argument(
        "--genwp", action="append", nargs=2, metavar=("LABEL", "CHECKPOINT"), default=[]
    )
    p.add_argument(
        "--fd-sanity",
        choices=["none", "first", "all"],
        default="none",
        help="Optional FD-vs-stored-GT implementation sanity check.",
    )
    p.add_argument("--fd-dx", type=float, default=10.0)
    p.add_argument("--fd-dz", type=float, default=10.0)
    p.add_argument("--fd-dt", type=float, default=0.001)
    p.add_argument("--fd-nt", type=int, default=1001)
    p.add_argument("--fd-freq", type=float, default=15.0)
    p.add_argument("--fd-pml", type=int, default=50)
    p.add_argument("--fd-downsample", type=int, default=10)
    return p.parse_args()


def load_case(path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path) as d:
            vp = np.asarray(d["vp"], dtype=np.float32)
            snaps = np.asarray(d["snaps"], dtype=np.float32)
            loc_x = int(np.asarray(d["loc_x"]).item())
    elif path.suffix.lower() == ".mat":
        d = sio.loadmat(path)
        vp = np.asarray(d["vp"], dtype=np.float32)
        snaps = np.asarray(d["snaps"], dtype=np.float32)
        loc_x = int(np.asarray(d["loc_x"]).squeeze().item())
    else:
        raise ValueError(f"Unsupported case type: {path}")

    if snaps.shape == (501, 128, 128):
        snaps = snaps[::5]
    elif snaps.shape != (101, 128, 128):
        raise ValueError(f"{path}: unexpected snaps shape {snaps.shape}")

    if vp.shape != (128, 128):
        raise ValueError(f"{path}: unexpected vp shape {vp.shape}")
    if not np.isfinite(vp).all() or not np.isfinite(snaps).all():
        raise ValueError(f"{path}: non-finite data")

    return vp, snaps, loc_x


def discover_cases(repo, selected):
    selected = selected or ["id", "marmousi_official", "marmousi_new", "redsea"]
    cases = []

    if "id" in selected:
        for name in ("Overthrust", "SEGEAGE"):
            for shot in (1, 2):
                p = repo / "dataset/test" / name / f"shot{shot}.mat"
                cases.append(("id", f"{name}_shot{shot}", p))

    if "marmousi_official" in selected:
        for shot in (1, 2):
            p = repo / "dataset/test/Marmousi" / f"shot{shot}.mat"
            cases.append(("marmousi_official", f"MarmousiOfficial_shot{shot}", p))

    if "marmousi_new" in selected:
        root = repo / "dataset/marmousi_ood_main_outputs"
        for p in sorted(root.glob("MarmousiMain*.npz")):
            cases.append(("marmousi_new", p.stem, p))

    if "redsea" in selected:
        root = repo / "dataset/redsea_ood_range_matched/outputs"
        for p in sorted(root.glob("RedSea_vel*_shot*.npz")):
            cases.append(("redsea", p.stem, p))

    missing = [str(p) for _, _, p in cases if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing cases:\n" + "\n".join(missing))
    if not cases:
        raise RuntimeError("No cases found")
    return cases


def import_genwp(repo):
    folder = str((repo / "genwp").resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)

    existing = sys.modules.get("code")
    if existing is not None and not hasattr(existing, "__path__"):
        del sys.modules["code"]

    module = importlib.import_module("code.script_util")
    if not Path(module.__file__).resolve().is_relative_to(repo.resolve()):
        raise RuntimeError("Wrong code.script_util imported")
    return module


def load_genwp(path, repo, device):
    util = import_genwp(repo)
    settings = util.model_and_diffusion_defaults()
    model, _ = util.create_model_and_diffusion(**settings, snap_steps=5)

    state = trusted_load(path)
    if isinstance(state, dict) and "ema_params" in state and "model" in state:
        raise ValueError("Pass GenWP EMA weights, not a full resume bundle")
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), settings.get("out_channels", 1)


def load_mcwp(path, device):
    checkpoint = trusted_load(path)
    kind = checkpoint.get("kind")
    config = ModelConfig(**checkpoint["model_config"])

    if kind in ("mcwp_ablation_full", "mcwp_ablation_ema"):
        model = build_model(config, checkpoint["variant"])
    elif kind in ("mcwp_full", "mcwp_ema"):
        model = MediumConditionedWavePropagator(config)
    else:
        raise ValueError(f"Unsupported MCWP checkpoint: {path} (kind={kind})")

    state = checkpoint["ema"] if kind.endswith("_full") else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    params = sum(x.numel() for x in model.parameters())
    return model, params


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    sync(device)
    t0 = time.perf_counter()
    value = fn()
    sync(device)
    return value, time.perf_counter() - t0


def runtime_stats(seconds):
    ms = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "runtime_mean_ms": float(ms.mean()),
        "runtime_median_ms": float(np.median(ms)),
        "runtime_std_ms": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
        "runtime_min_ms": float(ms.min()),
        "runtime_max_ms": float(ms.max()),
    }


def accuracy(pred, ref):
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    err = a - b

    num = np.sum(err[1:] ** 2)
    den = np.sum(b[1:] ** 2)

    return {
        "mae_pred100": float(np.abs(err[1:]).mean()),
        "relative_l2_pred100": float(np.sqrt(num / max(den, 1e-30))),
        "snr_db_pred100": float(
            10.0 * np.log10(max(den, 1e-30) / max(num, 1e-30))
        ),
        "final_frame_mae": float(np.abs(err[-1]).mean()),
    }


def frame_metrics(pred, ref):
    rows = []
    for n in range(101):
        a = pred[n].astype(np.float64)
        b = ref[n].astype(np.float64)
        err = a - b
        rows.append(
            {
                "frame": n,
                "mae": float(np.abs(err).mean()),
                "relative_l2": float(
                    np.sqrt(
                        np.sum(err ** 2)
                        / max(float(np.sum(b ** 2)), 1e-30)
                    )
                ),
            }
        )
    return rows


@torch.no_grad()
def mcwp_predict(model, vp_np, ref_np, device):
    vp = torch.as_tensor(vp_np, dtype=torch.float32, device=device)[None, None]
    u0 = torch.as_tensor(ref_np[0], dtype=torch.float32, device=device)
    medium = model.encode_medium(vp)

    history = torch.zeros((1, 5, 128, 128), dtype=torch.float32, device=device)
    history[:, 0] = u0

    pred = torch.empty((1, 101, 128, 128), dtype=torch.float32, device=device)
    pred[:, 0] = u0

    for n in range(1, 101):
        idx = torch.tensor([n], dtype=torch.long, device=device)
        nxt = model.predict_next(history, idx, medium)
        pred[:, n:n+1] = nxt
        history = torch.cat((nxt, history[:, :-1]), dim=1)

    return pred[0]


@torch.no_grad()
def genwp_predict(model, vp_np, ref_np, channels, device, seed):
    vp = torch.as_tensor(vp_np, dtype=torch.float32, device=device)[None, None]
    u0 = torch.as_tensor(ref_np[0], dtype=torch.float32, device=device)

    normalized_vp = 2.0 * (vp - 1400.0) / 3600.0 - 1.0
    diffusion_time = torch.full((1,), 0.999, dtype=torch.float32, device=device)

    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(
        (1, channels, 128, 128),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    history = torch.zeros((1, 5, 128, 128), dtype=torch.float32, device=device)
    history[:, 0] = u0

    pred = torch.empty((1, 101, 128, 128), dtype=torch.float32, device=device)
    pred[:, 0] = u0

    for n in range(1, 101):
        idx = torch.tensor([n], dtype=torch.long, device=device)
        cond = torch.cat((history, normalized_vp), dim=1)
        nxt = model(
            z,
            cond,
            idx.float() * 0.001,
            diffusion_time,
        )[:, :1]

        pred[:, n:n+1] = nxt
        history = torch.cat((nxt, history[:, :-1]), dim=1)

    return pred[0]


def import_fd(repo):
    folder = str((repo / "dataset/data_gen").resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from fd_torch import acoustic_iso_torch, FD10StaggeredConv2d, set_wavelet_torch
    return acoustic_iso_torch, FD10StaggeredConv2d, set_wavelet_torch


def build_fd(repo, vp_np, loc_x, device, args):
    acoustic_iso_torch, FD10StaggeredConv2d, set_wavelet_torch = import_fd(repo)
    rho = np.ones_like(vp_np, dtype=np.float32)

    solver = acoustic_iso_torch(
        dx=args.fd_dx,
        dz=args.fd_dz,
        dt=args.fd_dt,
        nt=args.fd_nt,
        loc_x_list=[int(loc_x)],
        freq=args.fd_freq,
        n_pml=args.fd_pml,
        vel=vp_np,
        rho=rho,
        device=str(device),
        dtype=torch.float32,
    )
    solver.set_method("FD10")
    solver.extend_wave_array()
    solver.build_boundary()

    src = set_wavelet_torch(
        freq=args.fd_freq,
        dt=args.fd_dt,
        nt=args.fd_nt,
        device=str(device),
        dtype=torch.float32,
    )

    fdop = FD10StaggeredConv2d(
        dx=args.fd_dx,
        dz=args.fd_dz,
        ext_cells=solver.ext_cells,
        device=str(device),
        dtype=torch.float32,
    ).to(device)

    return solver, src, fdop


@torch.no_grad()
def fd_predict(repo, vp_np, loc_x, device, args):
    solver, src, fdop = build_fd(repo, vp_np, loc_x, device, args)
    solver.set_source_loc_torch(int(loc_x))
    solver.reset_wavefields()

    nout = (solver.nt - 1) // args.fd_downsample + 1
    storage = torch.empty(
        (nout, solver.nz, solver.nx),
        dtype=solver.dtype,
        device=solver.device,
    )

    out_idx = 0

    for it in range(solver.nt):
        fdop.derx_backward_(solver.Vx, solver.dVx)
        fdop.derz_backward_(solver.Vz, solver.dVz)

        solver.acoustic_iso_stress_torch(it, src)
        solver.acoustic_iso_stress_plus_torch()

        fdop.derx_forward_(solver.pressure, solver.dpressure_Vx)
        fdop.derz_forward_(solver.pressure, solver.dpressure_Vz)

        solver.acoustic_velocity_torch()

        solver.sigmaxx0.copy_(solver.sigmaxx)
        solver.sigmazz0.copy_(solver.sigmazz)
        solver.Vx0.copy_(solver.Vx)
        solver.Vz0.copy_(solver.Vz)

        if it % args.fd_downsample == 0:
            storage[out_idx].copy_(solver.extract_matrix_torch())
            out_idx += 1

    return storage


def benchmark_prediction(fn, device, warmups, repeats):
    for _ in range(warmups):
        fn()
    sync(device)

    times = []
    first_pred = None

    for _ in range(repeats):
        out, seconds = timed_call(fn, device)
        times.append(seconds)
        if first_pred is None:
            first_pred = out.detach().cpu().numpy()

    return first_pred, runtime_stats(times)


def should_fd_sanity(mode, dataset_name, seen):
    if mode == "none":
        return False
    if mode == "all":
        return True
    return dataset_name not in seen


def main():
    args = get_args()

    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("Invalid warmups/repeats")

    repo = Path(args.repo).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cases = discover_cases(repo, args.dataset)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Cases: {len(cases)}")
    for group in sorted(set(x[0] for x in cases)):
        print(f"  {group}: {sum(x[0] == group for x in cases)}")

    jobs = []

    for label, checkpoint_text in args.mcwp:
        checkpoint = Path(checkpoint_text).expanduser().resolve()
        model, params = load_mcwp(checkpoint, device)
        jobs.append(
            {
                "kind": "mcwp",
                "label": label,
                "model": model,
                "params": params,
                "checkpoint": str(checkpoint),
            }
        )

    for label, checkpoint_text in args.genwp:
        checkpoint = Path(checkpoint_text).expanduser().resolve()
        model, channels = load_genwp(checkpoint, repo, device)
        jobs.append(
            {
                "kind": "genwp",
                "label": label,
                "model": model,
                "channels": channels,
                "params": sum(x.numel() for x in model.parameters()),
                "checkpoint": str(checkpoint),
            }
        )

    if not jobs:
        raise RuntimeError("Provide at least one --mcwp or --genwp")

    per_case_rows = []
    per_frame_rows = []
    fd_sanity_rows = []
    fd_sanity_seen = set()

    for case_idx, (dataset_name, case_name, case_path) in enumerate(cases):
        print(f"\n=== {dataset_name} | {case_name} ===", flush=True)

        vp, ref, loc_x = load_case(case_path)

        fd_fn = lambda: fd_predict(repo, vp, loc_x, device, args)
        fd_pred, fd_rt = benchmark_prediction(
            fd_fn, device, args.warmups, args.repeats
        )
        fd_median = fd_rt["runtime_median_ms"]

        print(f"FD10 median={fd_median:.3f} ms", flush=True)

        if should_fd_sanity(args.fd_sanity, dataset_name, fd_sanity_seen):
            fd_err = fd_pred.astype(np.float64) - ref.astype(np.float64)
            fd_acc = accuracy(fd_pred, ref)
            fd_sanity_rows.append(
                {
                    "dataset": dataset_name,
                    "case": case_name,
                    "mae_pred100": fd_acc["mae_pred100"],
                    "relative_l2_pred100": fd_acc["relative_l2_pred100"],
                    "max_abs": float(np.max(np.abs(fd_err))),
                }
            )
            fd_sanity_seen.add(dataset_name)

        # FD is the reference runtime; its accuracy is not included in learned-model summaries.
        per_case_rows.append(
            {
                "dataset": dataset_name,
                "case": case_name,
                "kind": "fd10",
                "label": "FD10",
                "parameters": 0,
                "checkpoint": "",
                "mae_pred100": 0.0,
                "relative_l2_pred100": 0.0,
                "snr_db_pred100": float("inf"),
                "final_frame_mae": 0.0,
                **fd_rt,
                "speedup_vs_fd": 1.0,
            }
        )

        for job in jobs:
            label = job["label"]
            model = job["model"]

            print(f"  {label} ...", end="", flush=True)

            if job["kind"] == "mcwp":
                run_fn = lambda m=model: mcwp_predict(m, vp, ref, device)
            else:
                case_seed = args.seed + case_idx * 100
                run_fn = (
                    lambda m=model, c=job["channels"], s=case_seed:
                    genwp_predict(m, vp, ref, c, device, s)
                )

            pred, rt = benchmark_prediction(
                run_fn, device, args.warmups, args.repeats
            )

            acc = accuracy(pred, ref)

            row = {
                "dataset": dataset_name,
                "case": case_name,
                "kind": job["kind"],
                "label": label,
                "parameters": job["params"],
                "checkpoint": job["checkpoint"],
                **acc,
                **rt,
                "speedup_vs_fd": fd_median / rt["runtime_median_ms"],
            }

            per_case_rows.append(row)

            for fm in frame_metrics(pred, ref):
                per_frame_rows.append(
                    {
                        "dataset": dataset_name,
                        "case": case_name,
                        "label": label,
                        **fm,
                    }
                )

            print(
                f" MAE={acc['mae_pred100']:.6g}"
                f" | relL2={acc['relative_l2_pred100']:.4f}"
                f" | runtime={rt['runtime_median_ms']:.2f} ms"
                f" | xFD={row['speedup_vs_fd']:.3f}",
                flush=True,
            )

    per_case_df = pd.DataFrame(per_case_rows)
    per_case_df.to_csv(out / "per_case.csv", index=False)

    per_frame_df = pd.DataFrame(per_frame_rows)
    per_frame_df.to_csv(out / "per_frame.csv", index=False)

    learned = per_case_df[per_case_df["kind"] != "fd10"].copy()

    summary_by_dataset = (
        learned.groupby(
            ["dataset", "label", "kind", "parameters"], as_index=False
        )
        .agg(
            cases=("case", "count"),
            mae_mean=("mae_pred100", "mean"),
            mae_std=("mae_pred100", "std"),
            relative_l2_mean=("relative_l2_pred100", "mean"),
            relative_l2_std=("relative_l2_pred100", "std"),
            snr_db_mean=("snr_db_pred100", "mean"),
            final_frame_mae_mean=("final_frame_mae", "mean"),
            runtime_median_ms_mean=("runtime_median_ms", "mean"),
            runtime_median_ms_std=("runtime_median_ms", "std"),
            speedup_vs_fd_mean=("speedup_vs_fd", "mean"),
        )
        .sort_values(["dataset", "mae_mean"])
    )

    summary_by_dataset.to_csv(out / "summary_by_dataset.csv", index=False)

    summary_overall = (
        learned.groupby(["label", "kind", "parameters"], as_index=False)
        .agg(
            cases=("case", "count"),
            mae_mean=("mae_pred100", "mean"),
            mae_std=("mae_pred100", "std"),
            relative_l2_mean=("relative_l2_pred100", "mean"),
            relative_l2_std=("relative_l2_pred100", "std"),
            snr_db_mean=("snr_db_pred100", "mean"),
            final_frame_mae_mean=("final_frame_mae", "mean"),
            runtime_median_ms_mean=("runtime_median_ms", "mean"),
            runtime_median_ms_std=("runtime_median_ms", "std"),
            speedup_vs_fd_mean=("speedup_vs_fd", "mean"),
        )
        .sort_values("mae_mean")
    )

    summary_overall.to_csv(out / "summary_overall.csv", index=False)

    fd_summary = (
        per_case_df[per_case_df["kind"] == "fd10"]
        .groupby("dataset", as_index=False)
        .agg(
            cases=("case", "count"),
            fd_runtime_median_ms_mean=("runtime_median_ms", "mean"),
            fd_runtime_median_ms_std=("runtime_median_ms", "std"),
        )
    )

    fd_summary.to_csv(out / "fd_runtime_summary.csv", index=False)

    if fd_sanity_rows:
        pd.DataFrame(fd_sanity_rows).to_csv(out / "fd_sanity.csv", index=False)

    protocol = {
        "repo": str(repo),
        "gpu": torch.cuda.get_device_name(device),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "datasets": args.dataset
        or ["id", "marmousi_official", "marmousi_new", "redsea"],
        "case_count": len(cases),
        "model_count": len(jobs),
        "accuracy": "learned prediction vs stored GT, forecast frames 1..100",
        "runtime": (
            "cold single-case CUDA latency; medium preprocessing included; "
            "checkpoint loading and disk I/O excluded"
        ),
        "fd_sanity": args.fd_sanity,
        "fd": {
            "dx": args.fd_dx,
            "dz": args.fd_dz,
            "dt": args.fd_dt,
            "nt": args.fd_nt,
            "freq": args.fd_freq,
            "pml": args.fd_pml,
            "downsample": args.fd_downsample,
        },
    }

    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))

    print("\n=== SUMMARY BY DATASET ===")
    print(summary_by_dataset.to_string(index=False))

    print("\n=== OVERALL LEARNED-MODEL SUMMARY ===")
    print(summary_overall.to_string(index=False))

    print("\n=== FD RUNTIME SUMMARY ===")
    print(fd_summary.to_string(index=False))

    if fd_sanity_rows:
        print("\n=== OPTIONAL FD SANITY ===")
        print(pd.DataFrame(fd_sanity_rows).to_string(index=False))

    print(f"\nSaved all results to: {out}")


if __name__ == "__main__":
    main()
