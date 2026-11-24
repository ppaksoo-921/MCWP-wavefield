"""Original study model loading and p0-only, 100-step evaluation.

No training, AMP, CUDA Graph, changed normalization, or future-FD injection.
The FD and autoregressive loops follow the supplied study evaluator.
"""
import gc
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from common import (DISPLAY, FD, VENDOR, array_sha, digest, jread, jwrite,
                    load_prepared, load_torch, module_at, save_npz,
                    select_cases, sha, verify_public)

PARAMETERS = dict(base=6307702, long100=6307702, fno=5371713,
                  genwp_published=125789825, genwp_retrained=125789825,
                  monolithic=6309446, static_fusion=6305654)
KNOWN_CHECKPOINT_HASHES = {
    "base": "5a2bafc1fe1d553dd69319b9195b490ea9dae63f0bccaf3dcc71e4db84a06e9e",
    "long100": "fd54d4c20534d0b5960058fa9f9441308eaaa0c46f2a31e6d8a4e1901141d15e",
    "fno": "9009d0fb3a425599f540915449aa9216e68c27d11f4011f4a93e4bdcd59f82a5",
    "genwp_published": "7f174ef7501ec6f15b457be3801e4b34b2390710f613521bc07d8082a58aecea",
}
NEURALOP_COMMIT = "8a0f5260983efffbe6e8ec39c2d095dba8cb2505"


def device_for(name):
    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Use the existing CUDA evaluation environment, e.g. conda activate fno-baseline.")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def environment(device):
    return dict(python=sys.version, os=platform.platform(), torch=torch.__version__,
                torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=str(device), gpu=torch.cuda.get_device_name(device),
                fp32=True, tf32=False, cudnn_benchmark=False, cuda_graph=False)


def fd_rollout(vp, loc, device):
    sys.path.insert(0, str(VENDOR))
    from fd_torch import acoustic_iso_torch, FD10StaggeredConv2d, set_wavelet_torch
    with torch.no_grad():
        solver = acoustic_iso_torch(dx=FD["dx_m"], dz=FD["dz_m"], dt=FD["dt_s"],
            nt=FD["nt"], loc_x_list=[int(loc)], freq=FD["frequency_hz"],
            n_pml=FD["pml_cells"], vel=vp, rho=np.ones_like(vp, dtype=np.float32),
            device=str(device), dtype=torch.float32)
        solver.set_method("FD10")
        solver.extend_wave_array()
        solver.build_boundary()
        src = set_wavelet_torch(freq=FD["frequency_hz"], dt=FD["dt_s"], nt=FD["nt"],
                                device=str(device), dtype=torch.float32)
        fdop = FD10StaggeredConv2d(dx=FD["dx_m"], dz=FD["dz_m"], ext_cells=solver.ext_cells,
                                  device=str(device), dtype=torch.float32).to(device)
        solver.set_source_loc_torch(int(loc))
        solver.reset_wavefields()
        storage = torch.empty((101, solver.nz, solver.nx), dtype=solver.dtype, device=solver.device)
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
            if it % FD["snapshot_stride"] == 0:
                storage[out_idx].copy_(solver.extract_matrix_torch())
                out_idx += 1
        return storage.cpu().numpy()


def load_reference(path):
    with np.load(path, allow_pickle=False) as d:
        vp = np.asarray(d["vp"], np.float32).copy()
        ref = np.asarray(d["snaps"], np.float32).copy()
        loc = int(np.asarray(d["loc_x"]).item())
    if ref.shape == (501, 128, 128):
        ref = ref[::5]
    if vp.shape != (128, 128) or ref.shape != (101, 128, 128):
        raise ValueError("Unexpected evaluation shape: " + str(path))
    if not np.isfinite(vp).all() or not np.isfinite(ref).all():
        raise ValueError("Non-finite reference: " + str(path))
    return vp, ref, loc


def fd_parity(c, m, device):
    first = m["cases"][0]
    signature = digest(dict(experiment=m["experiment_fingerprint"],
                            source=first["legacy_file_sha256"], fd=m["vendor_hashes"]["fd_torch.py"]))
    out = Path(c["work_dir"]) / "fd_solver_parity.json"
    if out.is_file():
        qa = jread(out)
        if qa.get("signature") == signature and qa.get("passed"):
            return
        raise RuntimeError("Existing FD parity record belongs to a different experiment.")
    vp, ref, loc = load_reference(first["legacy_file"])
    print("FD parity check on the PREVIOUS original-Marmousi first case ...", flush=True)
    pred = fd_rollout(vp, loc, device)
    delta = pred.astype(np.float64) - ref.astype(np.float64)
    max_abs = float(np.abs(delta).max())
    rel = float(np.sqrt(np.square(delta).sum() / max(float(np.square(ref.astype(np.float64)).sum()), 1e-30)))
    passed = bool(np.isfinite(pred).all() and max_abs <= 1e-4 and rel <= 1e-5)
    jwrite(out, dict(signature=signature, case=first["legacy_case"], max_abs=max_abs,
                     relative_l2=rel, passed=passed, environment=environment(device)))
    if not passed:
        raise RuntimeError("FD does not reproduce the previous reference. Stop before generating new cases: " + str(out))
    print("PASS FD parity: max_abs=%.4g, relL2=%.4g" % (max_abs, rel), flush=True)


def fd_signature(m, item):
    return digest(dict(experiment=m["experiment_fingerprint"], case=item["case"],
                       velocity=item["velocity_sha256"], loc_x=item["loc_x"], fd=FD))


def valid_fd(c, m, item):
    path = Path(c["work_dir"]) / item["file"]
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        return False
    saved = jread(sidecar)
    if saved["signature"] != fd_signature(m, item) or saved["file_sha256"] != sha(path):
        raise RuntimeError("Saved FD case differs from the current experiment: " + str(path))
    vp, ref, loc = load_reference(path)
    if array_sha(vp) != item["velocity_sha256"] or loc != item["loc_x"] or not np.any(ref):
        raise RuntimeError("Invalid saved FD case: " + str(path))
    return True


def generate(c, device_name, smoke=False):
    m = load_prepared(c)
    device = device_for(device_name)
    fd_parity(c, m, device)
    items = select_cases(m, smoke)
    for index, item in enumerate(items, 1):
        if valid_fd(c, m, item):
            print("FD reuse [%d/%d] %s" % (index, len(items), item["case"]), flush=True)
            continue
        path = Path(c["work_dir"]) / item["file"]
        with np.load(Path(c["work_dir"]) / item["velocity_file"], allow_pickle=False) as d:
            vp = d["vp"].copy()
        ref = fd_rollout(vp, item["loc_x"], device)
        if ref.shape != (101, 128, 128) or not np.isfinite(ref).all() or not np.any(ref):
            raise FloatingPointError("Invalid newly generated FD reference: " + item["case"])
        signature = fd_signature(m, item)
        save_npz(path, vp=vp, snaps=ref, loc_x=np.int64(item["loc_x"]),
                 velocity_unit=np.array("m/s"), dataset_group=np.array(item["group"]),
                 signature=np.array(signature), snapshot_times_s=np.arange(101) * .01)
        jwrite(path.with_suffix(".json"), dict(case=item["case"], signature=signature,
                file_sha256=sha(path), velocity_sha256=array_sha(vp), fd=FD,
                reference_sha256=array_sha(ref), environment=environment(device)))
        print("FD [%d/%d] %s" % (index, len(items), item["case"]), flush=True)
    verify_public(m["protected_genwp_public_files"])


def load_model(key, c, device):
    path = Path(c["checkpoints"][key])
    if not path.is_file():
        raise FileNotFoundError("Missing trained checkpoint for %s: %s" % (key, path))
    checksum = sha(path)
    if key in KNOWN_CHECKPOINT_HASHES and checksum != KNOWN_CHECKPOINT_HASHES[key]:
        raise RuntimeError("Checkpoint differs from the paper's recorded weights: " + str(path))
    bundle = load_torch(path)
    source = dict(checkpoint=str(path), checkpoint_sha256=checksum)
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    if key.startswith("genwp_"):
        repo = Path(c["genwp_repo"]).resolve()
        sys.path.insert(0, str(repo / "genwp"))
        existing = sys.modules.get("code")
        if existing is not None and not hasattr(existing, "__path__"):
            del sys.modules["code"]
        util = importlib.import_module("code.script_util")
        if not Path(util.__file__).resolve().is_relative_to(repo):
            raise RuntimeError("Wrong GenWP source imported: " + util.__file__)
        settings = util.model_and_diffusion_defaults()
        model, _ = util.create_model_and_diffusion(**settings, snap_steps=5)
        if "ema_params" in bundle or "optimizer" in bundle:
            raise ValueError("Pass GenWP evaluation EMA weights, not a resume checkpoint.")
        state = bundle
        if state and all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        source.update(genwp_source=str(util.__file__), genwp_source_sha256=sha(util.__file__),
                      noise_channels=settings.get("out_channels", 1))
    elif key == "fno":
        location = Path(c["neuraloperator_source"])
        if location.is_dir():
            sys.path.insert(0, str(location))
            git = subprocess.run(["git", "-C", str(location), "rev-parse", "HEAD"],
                                 capture_output=True, text=True)
            if git.returncode == 0 and git.stdout.strip() != NEURALOP_COMMIT:
                raise RuntimeError("Use the pinned NeuralOperator source from original FNO training.")
            if git.returncode == 0:
                source["neuraloperator_commit"] = git.stdout.strip()
        try:
            wrapper = module_at("_marmousi_fno_wrapper", VENDOR / "fno_wrapper.py")
        except ModuleNotFoundError as exc:
            raise RuntimeError("Activate the original fno-baseline environment; do not upgrade NeuralOperator. "
                               "Missing import: " + str(exc)) from exc
        if bundle.get("kind") != "fno_ema" or bundle.get("completed_steps") != 100000:
            raise ValueError("Expected the original FNO 100k EMA checkpoint.")
        signature = bundle["signature"]
        if (signature["neuraloperator_commit"] != NEURALOP_COMMIT or
                signature["implementation_sha256"]["fno_wrapper.py"] != sha(VENDOR / "fno_wrapper.py")):
            raise ValueError("FNO training implementation signature does not match.")
        model = wrapper.AutoregressiveFNO(wrapper.FNOConfig(**bundle["model_config"]))
        state = bundle["model"]
        import neuralop
        source["neuraloperator_imported_from"] = str(neuralop.__file__)
    else:
        from reference_model import ModelConfig, MediumConditionedWavePropagator
        from ablation_model import build_model
        cfg = ModelConfig(**bundle["model_config"])
        if (cfg.velocity_min, cfg.velocity_max) != (1400.0, 5000.0):
            raise ValueError("Keep the trained model's original 1400..5000 normalization.")
        kind = bundle.get("kind", "")
        if key in ("monolithic", "static_fusion"):
            if kind not in ("mcwp_ablation_ema", "mcwp_ablation_full") or bundle.get("variant") != key:
                raise ValueError("Wrong structural-control checkpoint for " + key)
            model = build_model(cfg, key)
        else:
            if kind not in ("mcwp_ema", "mcwp_full"):
                raise ValueError("Wrong MCWP checkpoint kind: " + kind)
            model = MediumConditionedWavePropagator(cfg)
        state = bundle["ema"] if kind.endswith("_full") else bundle["model"]
        source["model_config"] = bundle["model_config"]
    model.load_state_dict(state, strict=True)
    del bundle, state
    model = model.to(device).eval()
    count = sum(p.numel() for p in model.parameters())
    if count != PARAMETERS[key]:
        raise ValueError("Parameter count mismatch for %s: %d" % (key, count))
    source["parameters"] = count
    return model, source


def rollout(model, key, vp_np, ref_np, device, seed, channels=1):
    with torch.no_grad():
        vp = torch.as_tensor(vp_np, dtype=torch.float32, device=device)[None, None]
        u0 = torch.as_tensor(ref_np[0], dtype=torch.float32, device=device)
        history = torch.zeros((1, 5, 128, 128), dtype=torch.float32, device=device)
        history[:, 0] = u0
        pred = torch.empty((1, 101, 128, 128), dtype=torch.float32, device=device)
        pred[:, 0] = u0
        if key.startswith("genwp_"):
            normalized_vp = 2.0 * (vp - 1400.0) / 3600.0 - 1.0
            diffusion_time = torch.full((1,), .999, dtype=torch.float32, device=device)
            generator = torch.Generator(device=device).manual_seed(int(seed))
            z = torch.randn((1, channels, 128, 128), dtype=torch.float32,
                            device=device, generator=generator)
        else:
            medium = model.encode_medium(vp)
        for n in range(1, 101):
            idx = torch.tensor([n], dtype=torch.long, device=device)
            if key.startswith("genwp_"):
                cond = torch.cat((history, normalized_vp), dim=1)
                nxt = model(z, cond, idx.float() * .001, diffusion_time)[:, :1]
            else:
                nxt = model.predict_next(history, idx, medium)
            pred[:, n:n + 1] = nxt
            history = torch.cat((nxt, history[:, :-1]), dim=1)
        result = pred[0].cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite model prediction. No failed cases will be dropped from summaries.")
    return result


def metrics(pred, ref):
    p, r = pred.astype(np.float64), ref.astype(np.float64)
    e = p - r
    denom = float(np.square(r[1:]).sum())
    if denom <= 0:
        raise ValueError("Zero reference signal.")
    rows = []
    for n in range(1, 101):
        en, rn = e[n], r[n]
        rows.append(dict(frame=n, time_s=n / 100, mae=float(np.abs(en).mean()),
                         mse=float(np.square(en).mean()),
                         relative_l2=float(np.sqrt(np.square(en).sum() / max(float(np.square(rn).sum()), 1e-30)))))
    total = dict(mae=float(np.abs(e[1:]).mean()), mse=float(np.square(e[1:]).mean()),
                 relative_l2=float(np.sqrt(np.square(e[1:]).sum() / denom)),
                 final_frame_mae=float(np.abs(e[100]).mean()))
    return total, rows


def evaluate(c, device_name, smoke=False):
    m = load_prepared(c)
    device = device_for(device_name)
    folder = Path(c["work_dir"]) / ("smoke" if smoke else "evaluation")
    items = select_cases(m, smoke)
    for item in items:
        if not valid_fd(c, m, item):
            raise FileNotFoundError("Generate new FD references first: " + item["case"])
    for key in c["models"]:
        model, source = load_model(key, c, device)
        source["environment"] = environment(device)
        source["experiment_fingerprint"] = m["experiment_fingerprint"]
        source["initialization"] = "p0 only + four older zeros, free rollout through frame 100"
        jwrite(folder / key / "model_protocol.json", source)
        for index, item in enumerate(items, 1):
            fdpath = Path(c["work_dir"]) / item["file"]
            identity = digest(dict(experiment=m["experiment_fingerprint"], checkpoint=source["checkpoint_sha256"],
                case_sha256=sha(fdpath), seed=item["genwp_noise_seed"], fp32=True, tf32=False))
            output = folder / key / (item["case"] + ".json")
            if output.is_file():
                saved = jread(output)
                if saved.get("identity") == identity and saved.get("status") == "PASS":
                    print("Reuse %s [%d/%d] %s" % (DISPLAY[key], index, len(items), item["case"]), flush=True)
                    continue
                raise RuntimeError("Saved evaluation does not match current inputs: " + str(output))
            vp, ref, _ = load_reference(fdpath)
            start = time.perf_counter()
            pred = rollout(model, key, vp, ref, device, item["genwp_noise_seed"], source.get("noise_channels", 1))
            total, frames = metrics(pred, ref)
            if item["shot"] == 1:
                save_npz(folder / key / (item["case"] + "_snapshots.npz"),
                         frames=np.array([20, 50, 100]), prediction=pred[[20, 50, 100]],
                         reference=ref[[20, 50, 100]], vp=vp)
            jwrite(output, dict(identity=identity, status="PASS", model=key, label=DISPLAY[key],
                case=item["case"], legacy_case=item["legacy_case"], group=item["group"],
                region=item["region"], shot=item["shot"], loc_x=item["loc_x"],
                genwp_noise_seed=item["genwp_noise_seed"], checkpoint_sha256=source["checkpoint_sha256"],
                case_sha256=sha(fdpath), parameters=source["parameters"], metrics=total, per_frame=frames,
                environment=source["environment"],
                evaluation_elapsed_s=time.perf_counter() - start,
                timing_note="Diagnostic one-pass elapsed time, not a runtime benchmark."))
            print("%s [%d/%d] %s: MAE=%.8f, relL2=%.6f" %
                  (DISPLAY[key], index, len(items), item["case"], total["mae"], total["relative_l2"]), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    verify_public(m["protected_genwp_public_files"])
    print("Evaluation completed:", folder, flush=True)
