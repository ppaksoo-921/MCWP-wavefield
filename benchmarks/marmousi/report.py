"""Complete-coverage summaries and an explicitly labeled historical 66-case update."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
from common import (DISPLAY, HERE, digest, jread, jwrite, load_prepared, read_csv,
                    select_cases, sha, write_csv)


def summary(rows, group_by=()):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ("model",) + tuple(group_by))].append(r)
    result = []
    for key, selected in sorted(groups.items()):
        item = dict(zip(("model",) + tuple(group_by), key))
        item.update(label=DISPLAY[key[0]], cases=len(selected))
        for metric in ("mae", "mse", "relative_l2"):
            if all(metric in r for r in selected):
                values = [float(r[metric]) for r in selected]
                item[metric + "_mean"] = float(np.mean(values))
                item[metric + "_std_between_cases"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result.append(item)
    return result


def collect(c, m, smoke):
    folder = Path(c["work_dir"]) / ("smoke" if smoke else "evaluation")
    cases = select_cases(m, smoke)
    records, frames = [], []
    for model in c["models"]:
        protocol = jread(folder / model / "model_protocol.json")
        if protocol["experiment_fingerprint"] != m["experiment_fingerprint"]:
            raise RuntimeError("Model protocol belongs to a different experiment: " + model)
        for case in cases:
            p = folder / model / (case["case"] + ".json")
            if not p.is_file():
                raise RuntimeError("Incomplete coverage; no partial summaries: " + str(p))
            r = jread(p)
            if r["status"] != "PASS" or r["model"] != model or r["case"] != case["case"]:
                raise ValueError("Invalid result: " + str(p))
            if r["case_sha256"] != sha(Path(c["work_dir"]) / case["file"]):
                raise RuntimeError("Result/reference mismatch: " + str(p))
            identity = digest(dict(experiment=m["experiment_fingerprint"],
                checkpoint=protocol["checkpoint_sha256"], case_sha256=r["case_sha256"],
                seed=case["genwp_noise_seed"], fp32=True, tf32=False))
            if r["checkpoint_sha256"] != protocol["checkpoint_sha256"] or r["identity"] != identity:
                raise RuntimeError("Mixed checkpoint or evaluation identity: " + str(p))
            values = r["metrics"]
            if not all(np.isfinite(float(v)) and float(v) >= 0 for v in values.values()):
                raise ValueError("Invalid accuracy value: " + str(p))
            if [x["frame"] for x in r["per_frame"]] != list(range(1, 101)):
                raise ValueError("Need all future frames 1..100: " + str(p))
            if abs(np.mean([x["mae"] for x in r["per_frame"]]) - values["mae"]) > 1e-10:
                raise ValueError("Per-frame and whole-trajectory MAE disagree: " + str(p))
            records.append(dict(model=model, label=DISPLAY[model], group=case["group"],
                case=case["case"], legacy_case=case["legacy_case"], region=case["region"],
                shot=case["shot"], loc_x=case["loc_x"], parameters=r["parameters"],
                checkpoint_sha256=r["checkpoint_sha256"], **values))
            frames.extend(dict(model=model, label=DISPLAY[model], case=case["case"],
                               region=case["region"], **f) for f in r["per_frame"])
    return folder, records, frames


def historical_update(c, m, records, out):
    provenance = jread(HERE / "previous_reference_provenance.json")
    old_path = HERE / "previous_66_case_accuracy.csv"
    if sha(old_path) != provenance["bundled_csv_sha256"]:
        raise ValueError("Bundled historical reference changed.")
    expected = provenance["old_marmousi_case_file_sha256"]
    actual = {r["legacy_case"]: r["legacy_file_sha256"] for r in m["cases"]}
    if actual != expected:
        return {"available": False, "reason": "Legacy evaluation files differ from the historical paper records. "
                "New Marmousi accuracy remains valid; historical/macro update was intentionally omitted."}, []
    old = [r for r in read_csv(old_path) if r["model"] in c["models"]]
    for r in old:
        for key in ("mae", "relative_l2"):
            r[key] = float(r[key])
    lookup = {(r["model"], r["case"]): r for r in old if r["paper_group"] == "marmousi_new"}
    same_geometry = m.get("same_geometry_as_legacy_after_recorded_affine")
    comparison_type = "isolated_velocity_range_change" if same_geometry is True else "new_direct_raw_preparation_not_an_isolated_range_change"
    updated = [dict(r, origin="historical_unchanged") for r in old if r["paper_group"] != "marmousi_new"]
    paired = []
    for r in records:
        prior = lookup[(r["model"], r["legacy_case"])]
        change = dict(model=r["model"], label=r["label"], region=r["region"],
                      case=r["case"], legacy_case=r["legacy_case"], comparison_type=comparison_type)
        for key in ("mae", "relative_l2"):
            change["old_" + key] = prior[key]
            change["new_" + key] = r[key]
            change[key + "_change_percent"] = 100 * (r[key] / prior[key] - 1) if prior[key] > 0 else None
        paired.append(change)
        updated.append(dict(model=r["model"], label=r["label"], paper_group="marmousi_new",
                            case=r["case"], mae=r["mae"], relative_l2=r["relative_l2"],
                            origin="new_original_Marmousi_v1500_5000"))
    write_csv(out / "paired_previous_vs_new_marmousi.csv", paired)
    write_csv(out / "updated_66_per_case.csv", updated)
    group_means = []
    macro = []
    for model in c["models"]:
        data = [r for r in updated if r["model"] == model]
        sizes = {g: sum(r["paper_group"] == g for r in data) for g in provenance["group_counts"]}
        if len(data) != 66 or sizes != provenance["group_counts"]:
            raise ValueError("Historical replacement did not preserve the 6-group / 66-case design.")
        for group in provenance["group_counts"]:
            subset = [r for r in data if r["paper_group"] == group]
            group_means.append(dict(model=model, label=DISPLAY[model], paper_group=group,
                cases=len(subset), mae_mean=float(np.mean([r["mae"] for r in subset])),
                relative_l2_mean=float(np.mean([r["relative_l2"] for r in subset])),
                origin="new_original_Marmousi_v1500_5000" if group == "marmousi_new" else "historical_unchanged"))
        these = [r for r in group_means if r["model"] == model]
        prior = [r for r in old if r["model"] == model]
        old_group_mae = [np.mean([r["mae"] for r in prior if r["paper_group"] == g])
                         for g in provenance["group_counts"]]
        macro.append(dict(model=model, label=DISPLAY[model], cases=66, groups=6,
            mae_equal_group=float(np.mean([r["mae_mean"] for r in these])),
            relative_l2_equal_group=float(np.mean([r["relative_l2_mean"] for r in these])),
            mae_equal_case=float(np.mean([r["mae"] for r in data])),
            previous_mae_equal_group=float(np.mean(old_group_mae)),
            aggregation="Six evaluation-group means, weight 1/6 each; 15 new Marmousi cases, 51 historical cases"))
    write_csv(out / "updated_six_group_means.csv", group_means)
    write_csv(out / "updated_six_group_macro.csv", macro)
    return dict(available=True, new_cases=15, unchanged_historical_cases=51,
                genwp_public_marmousi_cases_unchanged=2,
                comparison_type=comparison_type,
                note="Recomputed accuracy on the new original-Marmousi group only; other groups were not rerun."), macro


def plots(c, m, records, frames, out, macro):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(base="#78B7B2", long100="#087E8B", fno="#718096", genwp_published="#8D6AB8",
                  genwp_retrained="#B99ACB", monolithic="#C2A26B", static_fusion="#B27575")
    models = c["models"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, region in zip(axes, ("Left", "Middle", "Right")):
        info = next(x for x in m["models"] if x["region"] == region)
        with np.load(Path(c["work_dir"]) / info["file"], allow_pickle=False) as d:
            im = ax.imshow(d["vp"], cmap="viridis", vmin=1500, vmax=5000)
        ax.set(title=region, xlabel="x grid index", ylabel="z grid index")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="P-wave velocity (m/s)", shrink=.8)
    fig.suptitle("Separately obtained original Marmousi: 1500-5000 m/s; not GenWP public Marmousi")
    fig.savefig(out / "velocity_models.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    means = [np.mean([r["mae"] for r in records if r["model"] == k]) for k in models]
    std = [np.std([r["mae"] for r in records if r["model"] == k], ddof=1) for k in models]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=std, capsize=3, color=[colors[k] for k in models])
    ax.set_xticks(x, [DISPLAY[k].replace("GenWP-", "GenWP\n") for k in models], rotation=20, ha="right")
    ax.set(ylabel="Trajectory MAE", title="Original Marmousi, 1500-5000 m/s: mean and SD across evaluated cases")
    fig.tight_layout(); fig.savefig(out / "marmousi_model_comparison.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for key in models:
        selected = [r for r in frames if r["model"] == key]
        y = [np.mean([r["mae"] for r in selected if r["frame"] == n]) for n in range(1, 101)]
        ax.plot(np.arange(1, 101) * .01, y, label=DISPLAY[key], color=colors[key])
    ax.set(xlabel="Prediction time (s)", ylabel="Mean MAE", title="Original Marmousi: full free-rollout error")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "mae_over_time.png", dpi=180); plt.close(fig)
    if macro:
        lookup = {r["model"]: r for r in macro}
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - .19, [lookup[k]["previous_mae_equal_group"] for k in models], .38,
               color="#D4D8DD", label="Previous 66-case reference")
        ax.bar(x + .19, [lookup[k]["mae_equal_group"] for k in models], .38,
               color=[colors[k] for k in models], label="Replace original Marmousi with 1500-5000")
        ax.set_xticks(x, [DISPLAY[k].replace("GenWP-", "GenWP\n") for k in models], rotation=20, ha="right")
        ax.set(ylabel="Equal-group mean MAE", title="Six evaluation groups, equal weights (1/6 each)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "updated_six_group_macro.png", dpi=180); plt.close(fig)


def report(c, smoke=False):
    m = load_prepared(c)
    folder, records, frames = collect(c, m, smoke)
    out = folder / "report"
    out.mkdir(parents=True, exist_ok=True)
    jwrite(out / "effective_config.json", c)
    write_csv(out / "per_case.csv", records)
    write_csv(out / "per_frame.csv", frames)
    means = summary(records)
    write_csv(out / "summary_original_marmousi.csv", means)
    write_csv(out / "summary_by_region.csv", summary(records, ("region",)))
    historical, macro = ({"available": False, "reason": "SMOKE only; no full-group or 66-case claims."}, [])
    if not smoke:
        historical, macro = historical_update(c, m, records, out)
    plots(c, m, records, frames, out, macro)
    protocol = dict(experiment_fingerprint=m["experiment_fingerprint"], smoke_only=smoke,
        source_mode=m.get("source_mode"), actual_velocity_source=m.get("raw_binary_audit"),
        evaluated_cases=len(select_cases(m, smoke)), evaluated_models=c["models"],
        accuracy_frames="1..100, .01..1.00 s; frame 0 excluded", historical_update=historical,
        public_marmousi="GenWP public shot1/shot2 are not rescaled or regenerated.",
        runtime="This package reports accuracy, not benchmark speedup.",
        standard_deviation="Variation across evaluated source cases, not across independent training seeds.")
    jwrite(out / "report_protocol.json", protocol)
    for r in means:
        print("%-18s MAE=%.8f  relL2=%.6f  n=%d" %
              (r["label"], r["mae_mean"], r["relative_l2_mean"], r["cases"]))
    name = "marmousi_v1500_5000_%sreport_%s.zip" % ("SMOKE_" if smoke else "", datetime.now().strftime("%Y%m%d_%H%M%S"))
    zip_path = Path(c["work_dir"]) / name
    # Small report only: never include weights or pressure/prediction arrays.
    files = list(out.glob("*"))
    files += [Path(c["work_dir"]) / "preparation.json", Path(c["work_dir"]) / "fd_solver_parity.json"]
    files += [Path(c["work_dir"]) / "raw_original_preview.png"]
    files += [p for p in (Path(c["work_dir"]) / "fd_cases").glob("*.json")]
    files += [p for p in (Path(c["work_dir"]) / "source_audit").glob("*.json")]
    files += [p for key in c["models"] for p in (folder / key).glob("*.json")]
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
        for path in files:
            if path.is_file():
                z.write(path, str(path.relative_to(Path(c["work_dir"]))))
        for name in ("config.json", "source_metadata_reference.json", "previous_reference_provenance.json"):
            z.write(HERE / name, "package/" + name)
    print("Report ZIP (no weights/FD arrays):", zip_path, flush=True)
    return zip_path
