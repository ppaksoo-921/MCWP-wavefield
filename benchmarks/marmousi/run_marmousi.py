#!/usr/bin/env python3
"""Original Marmousi -> 1500..5000 m/s -> new FD -> seven trained models."""
import argparse
from pathlib import Path
import sys
from common import DISPLAY, HERE, REGIONS, SHOTS, bootstrap, config


def plan(c):
    print("Experiment: separately obtained ORIGINAL Marmousi, not GenWP public Marmousi")
    print("Original physical velocities: 1500..5500 m/s")
    print("New physical velocities:      1500..5000 m/s (minimum is preserved)")
    print("Model input normalization:    1400..5000 m/s (unchanged)")
    print("ACTUAL velocity input:", c["original_binary"])
    print("Binary layout:", c.get("raw_binary_layout", "xz_C"), "; no main/hard inverse reconstruction")
    print("Geometry: full original file -> 3 raw 751x767 windows -> 128x128 linear resize")
    print("Sources:", list(SHOTS), "; total cases: 15; FD: 15 Hz / 10 m / .001 s / 1001 updates")
    print("All new output:", c["work_dir"])
    checks = [Path(c["metadata"]), Path(c["original_binary"])]
    checks += [Path(c["legacy_cases_dir"]) / ("MarmousiMain%s_vel1_shot%d.npz" % (r, s))
               for r in REGIONS for s in range(1, 6)]
    checks += [Path(c["checkpoints"][k]) for k in c["models"]]
    missing = [str(p) for p in checks if not p.is_file()]
    for k in c["models"]:
        print("  %s: %s" % (DISPLAY[k], "FOUND" if Path(c["checkpoints"][k]).is_file() else "MISSING"))
    if missing:
        raise FileNotFoundError("Update config.json if paths moved:\n" + "\n".join(missing))
    print("PLAN PASS. No GPU used. No training or source-file changes.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "prepare", "generate", "evaluate", "report", "smoke", "all"))
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda:0", help="Physical GPU in an unmasked shell, e.g. cuda:8")
    p.add_argument("--models", nargs="+", choices=list(DISPLAY), default=None,
                   help="Optional explicit subset; default is all seven paper models.")
    p.add_argument("--smoke", action="store_true", help="Only first shot of each original region; never updates 66-case summaries.")
    args = p.parse_args()
    bootstrap()
    c = config(args.config)
    if args.models:
        c["models"] = args.models
    if args.stage == "plan":
        plan(c)
        return
    if args.stage in ("prepare", "smoke", "all"):
        from prepare import prepare
        prepare(c)
    if args.stage in ("generate", "smoke", "all"):
        from inference import generate
        generate(c, args.device, args.smoke or args.stage == "smoke")
    if args.stage in ("evaluate", "smoke", "all"):
        from inference import evaluate
        evaluate(c, args.device, args.smoke or args.stage == "smoke")
    if args.stage in ("report", "smoke", "all"):
        from report import report
        report(c, args.smoke or args.stage == "smoke")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Completed cases are retained and verified on resume.", file=sys.stderr)
        raise SystemExit(130)
