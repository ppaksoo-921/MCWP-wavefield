#!/usr/bin/env python3

"""
Official NeuralOperator FNO baseline training.

The data sampling, rollout horizon, causal weighting, optimizer,
EMA, global batch, and random seed match the MCWP Base training.

FNO core:
    neuraloperator/neuraloperator
    commit 8a0f5260983efffbe6e8ec39c2d095dba8cb2505
"""

import argparse
import csv
import fcntl
import os
import random
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


MCWP_DIR = Path.home() / "Wage_surrogate/mcwp"

if str(MCWP_DIR) not in sys.path:
    sys.path.append(str(MCWP_DIR))

from common import (
    CausalWeights,
    all_gather_tensor,
    atomic_save,
    cpu_tree,
    get_rng,
    set_rng,
    trusted_load,
    hash_file,
    write_json,
)

from data import WaveDataset, ResumableLoader
from fno_wrapper import FNOConfig, AutoregressiveFNO


STOP_REQUESTED = False

NEURALOP_COMMIT = (
    "8a0f5260983efffbe6e8ec39c2d095dba8cb2505"
)


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parser():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--data-dir",
        default=str(
            Path.home()
            / "Wage_surrogate/GenWP-pub/"
              "dataset/phaseA_outputs/h5"
        ),
    )

    p.add_argument("--run-dir", required=True)

    p.add_argument(
        "--max-steps",
        type=int,
        default=100000,
    )

    p.add_argument(
        "--global-batch",
        type=int,
        default=36,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--width",
        type=int,
        default=96,
    )

    p.add_argument(
        "--modes",
        type=int,
        default=12,
    )

    p.add_argument(
        "--layers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--horizon",
        type=int,
        default=3,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--ema",
        type=float,
        default=0.999,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260909,
    )

    p.add_argument(
        "--no-causal",
        action="store_true",
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--eval-every",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--eval-at",
        default="1000",
    )

    p.add_argument(
        "--keep-last",
        type=int,
        default=2,
    )

    p.add_argument(
        "--preserve-full",
        default="20000,50000,100000",
    )

    p.add_argument(
        "--resume",
        default="",
        help="empty/fresh, auto, or explicit full checkpoint",
    )

    p.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=0,
    )

    return p


def int_set(text):
    return {
        int(x)
        for x in text.split(",")
        if x.strip()
    }


def broadcast_message(message, world):
    if world > 1:
        payload = [message]
        dist.broadcast_object_list(
            payload,
            src=0,
        )
        return payload[0]

    return message


def main():
    global STOP_REQUESTED

    args = parser().parse_args()

    if args.horizon < 1:
        raise ValueError(
            "horizon must be positive"
        )

    if min(
        args.max_steps,
        args.global_batch,
        args.log_every,
        args.save_every,
        args.eval_every,
        args.keep_last,
    ) < 1:
        raise ValueError(
            "Step intervals and batch "
            "must be positive."
        )

    rank = int(
        os.environ.get("RANK", 0)
    )

    world = int(
        os.environ.get("WORLD_SIZE", 1)
    )

    local = int(
        os.environ.get(
            "LOCAL_RANK",
            args.local_rank,
        )
    )

    if args.global_batch % world:
        raise ValueError(
            "Global batch must be evenly "
            "divisible by GPU count."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        f"cuda:{local}"
    )

    torch.cuda.set_device(local)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    if world > 1:
        dist.init_process_group(
            "nccl",
            timeout=timedelta(minutes=20),
        )

    run = (
        Path(args.run_dir)
        .expanduser()
        .resolve()
    )

    run.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock = None
    lock_error = None

    if rank == 0:
        try:
            lock = open(
                run / ".train.lock",
                "a+",
            )

            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX
                | fcntl.LOCK_NB,
            )

        except Exception as exc:
            lock_error = (
                f"Another training process owns "
                f"{run}, or lock failed: {exc}"
            )

    lock_error = broadcast_message(
        lock_error,
        world,
    )

    if lock_error:
        raise RuntimeError(lock_error)

    data = None

    try:
        full_dir = run / "checkpoints"
        eval_dir = run / "eval_checkpoints"

        full_dir.mkdir(exist_ok=True)
        eval_dir.mkdir(exist_ok=True)

        resume_path = None

        if rank == 0:
            if args.resume == "auto":
                candidates = sorted(
                    full_dir.glob(
                        "resume_*.pt"
                    )
                )

                resume_path = (
                    str(candidates[-1])
                    if candidates
                    else None
                )

            elif args.resume not in (
                "",
                "fresh",
            ):
                resume_path = str(
                    Path(args.resume)
                    .expanduser()
                    .resolve()
                )

        resume_path = broadcast_message(
            resume_path,
            world,
        )

        if resume_path is None:
            if (
                list(
                    full_dir.glob(
                        "resume_*.pt"
                    )
                )
                or list(
                    eval_dir.glob(
                        "ema_*.pt"
                    )
                )
            ):
                raise RuntimeError(
                    "Run already has checkpoints. "
                    "Use --resume auto or a NEW "
                    "run directory."
                )

        bundle = (
            trusted_load(resume_path)
            if resume_path
            else None
        )

        if (
            bundle
            and bundle.get("kind")
            != "fno_full"
        ):
            raise ValueError(
                "Resume requires an FNO "
                "full checkpoint."
            )

        completed = (
            int(
                bundle[
                    "completed_steps"
                ]
            )
            if bundle
            else 0
        )

        if completed >= args.max_steps:
            if rank == 0:
                print(
                    f"Already at {completed} "
                    f"completed updates; "
                    f"target={args.max_steps}.",
                    flush=True,
                )

            return

        model_config = FNOConfig(
            modes=args.modes,
            width=args.width,
            layers=args.layers,
            history=5,
            velocity_min=1400.0,
            velocity_max=5000.0,
        )

        signature = {
            "architecture":
                "official_neuraloperator_fno",

            "neuraloperator_commit":
                NEURALOP_COMMIT,

            "model":
                asdict(model_config),

            "input_channels": 9,

            "input_definition": [
                "5_history_wavefields",
                "normalized_vp",
                "target_time",
                "x_coordinate",
                "z_coordinate",
            ],

            "global_batch":
                args.global_batch,

            "world_size":
                world,

            "horizon":
                args.horizon,

            "lr":
                args.lr,

            "weight_decay":
                args.weight_decay,

            "ema":
                args.ema,

            "seed":
                args.seed,

            "causal":
                not args.no_causal,

            "grad_clip":
                args.grad_clip,

            "precision":
                "float32",

            "tf32":
                False,
        }

        dataset = WaveDataset(
            args.data_dir,
            horizon=args.horizon,
        )

        signature[
            "dataset_index_hash"
        ] = dataset.index_hash

        signature[
            "dataset_count"
        ] = len(dataset)

        manifest = (
            Path(args.data_dir)
            .expanduser()
            .resolve()
            .parents[1]
            / "phaseA_inputs/"
              "phaseA_manifest.csv"
        )

        signature[
            "source_manifest_sha256"
        ] = (
            hash_file(manifest)
            if manifest.exists()
            else None
        )

        signature[
            "implementation_sha256"
        ] = {
            "fno_wrapper.py":
                hash_file(
                    Path(__file__).parent
                    / "fno_wrapper.py"
                ),

            "train_fno.py":
                hash_file(
                    Path(__file__)
                ),

            "mcwp/data.py":
                hash_file(
                    MCWP_DIR / "data.py"
                ),

            "mcwp/common.py":
                hash_file(
                    MCWP_DIR / "common.py"
                ),
        }

        if (
            bundle
            and bundle["signature"]
            != signature
        ):
            differing = [
                k
                for k in signature
                if signature[k]
                != bundle[
                    "signature"
                ].get(k)
            ]

            raise ValueError(
                "Resume configuration/data/code "
                "mismatch: "
                + ", ".join(differing)
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        raw = AutoregressiveFNO(
            model_config
        ).to(device)

        if bundle:
            raw.load_state_dict(
                bundle["model"],
                strict=True,
            )

        model = (
            DistributedDataParallel(
                raw,
                device_ids=[local],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if world > 1
            else raw
        )

        opt = torch.optim.AdamW(
            raw.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        ema = {
            k: v.detach().clone()
            for k, v
            in raw.state_dict().items()
        }

        causal = CausalWeights(device)

        if bundle:
            opt.load_state_dict(
                bundle["optimizer"]
            )

            for name, value in (
                bundle["ema"].items()
            ):
                ema[name].copy_(
                    value.to(device)
                )

            causal.load_state_dict(
                bundle["causal"]
            )

        data = ResumableLoader(
            dataset,
            args.global_batch,
            rank,
            world,
            args.seed,
            args.workers,
            completed,
            True,
        )

        random.seed(
            args.seed + rank + 1
        )

        np.random.seed(
            args.seed + rank + 1
        )

        torch.manual_seed(
            args.seed + rank + 1
        )

        if bundle:
            set_rng(
                bundle[
                    "rng_by_rank"
                ][rank],
                device,
            )

        del bundle

        if rank == 0:
            version = {
                "torch":
                    torch.__version__,

                "python":
                    sys.version,

                "numpy":
                    np.__version__,

                "device":
                    str(device),

                "gpu":
                    torch.cuda.get_device_name(
                        device
                    ),
            }

            if not (
                run / "config.json"
            ).exists():
                write_json(
                    {
                        "signature":
                            signature,

                        "runtime":
                            version,

                        "argv":
                            sys.argv,

                        "created":
                            datetime.now()
                            .isoformat(),
                    },
                    run / "config.json",
                )

            params = sum(
                p.numel()
                for p
                in raw.parameters()
            )

            print(
                "Architecture="
                "Official NeuralOperator FNO",
                flush=True,
            )

            print(
                f"modes={args.modes} "
                f"| width={args.width} "
                f"| layers={args.layers} "
                f"| parameters={params:,}",
                flush=True,
            )

            print(
                f"World={world} "
                f"| local_batch="
                f"{args.global_batch // world} "
                f"| global_batch="
                f"{args.global_batch} "
                f"| FP32",
                flush=True,
            )

            print(
                f"Trajectories="
                f"{len(dataset):,} "
                f"| steps/epoch="
                f"{data.steps_per_epoch} "
                f"| training horizon="
                f"{args.horizon}",
                flush=True,
            )

            print(
                f"Start completed updates="
                f"{completed} "
                f"| target="
                f"{args.max_steps} "
                f"| resume="
                f"{resume_path}",
                flush=True,
            )

        signal.signal(
            signal.SIGTERM,
            request_stop,
        )

        signal.signal(
            signal.SIGINT,
            request_stop,
        )

        full_keep = int_set(
            args.preserve_full
        )

        eval_at = int_set(
            args.eval_at
        )

        metric_path = (
            run / "metrics.csv"
        )

        fields = [
            "step",
            "samples",
            "supervised_targets_in_log_window",
            "total_loss",
            "data_loss",
            "mse_one_step",
            "mse_rollout",
            "grad_norm",
            "causal_weight_min",
            "seconds_per_step",
            "max_peak_allocated_gib",
        ]

        sums = np.zeros(
            5,
            dtype=np.float64,
        )

        intervals = 0

        clock = time.monotonic()
        started = clock

        torch.cuda.reset_peak_memory_stats(
            device
        )

        raw.train()

        def save(full, evaluation):
            rngs = None

            if full:
                local_rng = get_rng(
                    device
                )

                if world > 1:
                    rngs = [
                        None
                    ] * world

                    dist.all_gather_object(
                        rngs,
                        local_rng,
                    )

                else:
                    rngs = [
                        local_rng
                    ]

            error = None

            if rank == 0:
                try:
                    meta = {
                        "format_version":
                            1,

                        "completed_steps":
                            completed,

                        "model_config":
                            asdict(
                                model_config
                            ),

                        "signature":
                            signature,
                    }

                    if full:
                        checkpoint = dict(
                            meta,

                            kind="fno_full",

                            model=cpu_tree(
                                raw.state_dict()
                            ),

                            ema=cpu_tree(
                                ema
                            ),

                            optimizer=cpu_tree(
                                opt.state_dict()
                            ),

                            causal=cpu_tree(
                                causal.state_dict()
                            ),

                            rng_by_rank=rngs,

                            sampler_epoch=
                                completed
                                // data.steps_per_epoch,

                            sampler_batch_offset=
                                completed
                                % data.steps_per_epoch,
                        )

                        dst = (
                            full_dir
                            / f"resume_"
                              f"{completed:06d}.pt"
                        )

                        atomic_save(
                            checkpoint,
                            dst,
                        )

                        del checkpoint

                        print(
                            "Saved full: "
                            + str(dst),
                            flush=True,
                        )

                        files = sorted(
                            full_dir.glob(
                                "resume_*.pt"
                            )
                        )

                        newest = set(
                            files[
                                -args.keep_last:
                            ]
                        )

                        for path in files:
                            step = int(
                                path.stem
                                .split("_")[-1]
                            )

                            if (
                                path
                                not in newest
                                and step
                                not in full_keep
                            ):
                                path.unlink()

                    if evaluation:
                        dst = (
                            eval_dir
                            / f"ema_"
                              f"{completed:06d}.pt"
                        )

                        atomic_save(
                            dict(
                                meta,
                                kind="fno_ema",
                                model=cpu_tree(
                                    ema
                                ),
                            ),
                            dst,
                        )

                        print(
                            "Saved EMA: "
                            + str(dst),
                            flush=True,
                        )

                except Exception as exc:
                    error = (
                        "Checkpoint save failed: "
                        f"{exc}"
                    )

            error = broadcast_message(
                error,
                world,
            )

            if error:
                raise RuntimeError(error)

        while (
            completed
            < args.max_steps
        ):
            batch = next(data)

            batch = {
                k: v.to(
                    device,
                    non_blocking=True,
                )
                for k, v
                in batch.items()
            }

            prediction = model(
                batch["history"],
                batch["vp"],
                batch["index"],
                args.horizon,
            )

            squared = (
                prediction
                - batch["target"]
            ).square().mean(
                dim=(-2, -1)
            )

            valid = batch["valid"]

            indices = (
                batch["index"][:, None]
                + torch.arange(
                    args.horizon,
                    device=device,
                )[None]
            )

            table = (
                causal.weights()
                if not args.no_causal
                else torch.ones(
                    101,
                    device=device,
                )
            )

            weights = table[
                indices.clamp_max(100)
            ]

            per_sample = (
                squared
                * valid
                * weights
            ).sum(1) / (
                valid.sum(1)
                .clamp_min(1)
            )

            data_loss = (
                per_sample.mean()
            )

            total = data_loss

            ok = torch.isfinite(
                total
            ).to(
                dtype=torch.int32
            )

            if world > 1:
                dist.all_reduce(
                    ok,
                    op=dist.ReduceOp.MIN,
                )

            if not bool(ok.item()):
                raise FloatingPointError(
                    "Non-finite loss at "
                    f"update {completed + 1}"
                )

            opt.zero_grad(
                set_to_none=True
            )

            total.backward()

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    raw.parameters(),
                    args.grad_clip
                    if args.grad_clip > 0
                    else float("inf"),
                )
            )

            if not torch.isfinite(
                grad_norm
            ).item():
                raise FloatingPointError(
                    "Non-finite gradients "
                    f"at update "
                    f"{completed + 1}"
                )

            opt.step()

            with torch.no_grad():
                for k, v in (
                    raw.state_dict()
                    .items()
                ):
                    if v.is_floating_point():
                        ema[k].mul_(
                            args.ema
                        ).add_(
                            v,
                            alpha=
                                1
                                - args.ema,
                        )
                    else:
                        ema[k].copy_(v)

                if not args.no_causal:
                    ids = all_gather_tensor(
                        batch[
                            "index"
                        ].detach()
                    )

                    err = all_gather_tensor(
                        squared[
                            :, 0
                        ].detach()
                    )

                    causal.update(
                        ids,
                        err,
                    )

            completed += 1

            stats = torch.stack(
                (
                    total.detach(),
                    data_loss.detach(),
                    squared[
                        :, 0
                    ].detach().mean(),
                    (
                        (
                            squared
                            * valid
                        ).sum(1)
                        / valid.sum(1)
                        .clamp_min(1)
                    )
                    .detach()
                    .mean(),
                    grad_norm.detach(),
                )
            )

            if world > 1:
                dist.all_reduce(
                    stats
                )

                stats /= world

            sums += (
                stats
                .cpu()
                .numpy()
            )

            intervals += 1

            stop = torch.tensor(
                int(
                    STOP_REQUESTED
                ),
                device=device,
            )

            if world > 1:
                dist.all_reduce(
                    stop,
                    op=dist.ReduceOp.MAX,
                )

            final = (
                completed
                == args.max_steps
                or bool(
                    stop.item()
                )
            )

            if (
                completed
                % args.log_every
                == 0
                or final
            ):
                torch.cuda.synchronize(
                    device
                )

                now = time.monotonic()

                peak = torch.tensor(
                    torch.cuda
                    .max_memory_allocated(
                        device
                    )
                    / 1024**3,
                    device=device,
                )

                if world > 1:
                    dist.all_reduce(
                        peak,
                        op=dist.ReduceOp.MAX,
                    )

                if rank == 0:
                    avg = (
                        sums
                        / intervals
                    )

                    causal_min = (
                        causal.weights()
                        .min()
                        .item()
                        if not args.no_causal
                        else 1.0
                    )

                    row = [
                        completed,
                        completed
                        * args.global_batch,
                        int(
                            intervals
                            * args.global_batch
                            * args.horizon
                        ),
                        avg[0],
                        avg[1],
                        avg[2],
                        avg[3],
                        avg[4],
                        causal_min,
                        (
                            now
                            - clock
                        )
                        / intervals,
                        peak.item(),
                    ]

                    header = (
                        not metric_path.exists()
                    )

                    with open(
                        metric_path,
                        "a",
                        newline="",
                    ) as f:
                        writer = csv.writer(f)

                        if header:
                            writer.writerow(
                                fields
                            )

                        writer.writerow(
                            row
                        )

                    print(
                        f"Step "
                        f"{completed}/"
                        f"{args.max_steps} "
                        f"| data="
                        f"{avg[1]:.6g} "
                        f"| one_step_mse="
                        f"{avg[2]:.6g} "
                        f"| rollout_mse="
                        f"{avg[3]:.6g} "
                        f"| grad="
                        f"{avg[4]:.4g} "
                        f"| sec/step="
                        f"{(now-clock)/intervals:.3f} "
                        f"| peak="
                        f"{peak.item():.2f} GiB",
                        flush=True,
                    )

                sums[:] = 0
                intervals = 0
                clock = now

            do_full = (
                completed
                % args.save_every
                == 0
                or final
            )

            do_eval = (
                completed
                % args.eval_every
                == 0
                or completed
                in eval_at
                or final
            )

            if do_full or do_eval:
                save(
                    do_full,
                    do_eval,
                )

            if final:
                break

        if rank == 0:
            print(
                "Training "
                f"{'stopped on signal' if STOP_REQUESTED else 'finished'} "
                f"at {completed} updates. "
                f"Elapsed seconds: "
                f"{time.monotonic()-started:.2f}",
                flush=True,
            )

    finally:
        if data is not None:
            data.close()

        if dist.is_initialized():
            dist.destroy_process_group()

        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
