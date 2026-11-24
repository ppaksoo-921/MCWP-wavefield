"""CPU-only checks of distributed inputs, reported tables, and weights."""

import csv
import hashlib
import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


class RepositoryChecks(unittest.TestCase):
    def test_checkpoints_match_final_study(self):
        known = {
            "mcwp_final_ema_004000.pt": "fd54d4c20534d0b5960058fa9f9441308eaaa0c46f2a31e6d8a4e1901141d15e",
            "mcwp_base_ema_100000.pt": "5a2bafc1fe1d553dd69319b9195b490ea9dae63f0bccaf3dcc71e4db84a06e9e",
            "fno_ema_100000.pt": "9009d0fb3a425599f540915449aa9216e68c27d11f4011f4a93e4bdcd59f82a5",
            "monolithic_ema_100000.pt": "57351497ec4a54f3a9ed13b7089b1ebb200e24553e81ede6ac72ec30c7714511",
            "static_fusion_ema_100000.pt": "7cd5586bde825daf07e4844bbe8b5fbb421f350767c715c07749d1ac1136b674",
        }
        manifest = json.loads((ROOT / "weights/manifest.json").read_text())
        for name, expected in known.items():
            with self.subTest(name=name):
                self.assertEqual(manifest[name]["sha256"], expected)
                self.assertEqual(sha_file(ROOT / "weights" / name), expected)

    def test_final66_manifest_and_supplement_agree(self):
        cases = json.loads((ROOT / "benchmarks/cache/paper_cases_final66.json").read_text())["cases"]
        with (ROOT / "results/supplement/evaluation_cases.csv").open(newline="") as stream:
            supplement = list(csv.DictReader(stream))
        self.assertEqual(len(cases), 66)
        self.assertEqual(len(supplement), 66)
        self.assertEqual(Counter(x["group"] for x in cases),
                         {"ID": 4, "Marmousi_GenWP": 2, "Marmousi_source": 15,
                          "RedSea": 20, "Kimberlina": 10, "Pluto": 15})
        self.assertEqual({x["case"]: x["expected_sha256"] for x in cases},
                         {x["case"]: x["case_file_sha256"] for x in supplement})

    def test_redsea_source_to_paper_velocity(self):
        import numpy as np
        path = ROOT / "datasets/redsea/prepare_redsea.py"
        spec = importlib.util.spec_from_file_location("redsea_prepare", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prepared = module.prepare()
        self.assertEqual(len(prepared), 4)
        for i, vp in prepared.items():
            self.assertEqual(vp.shape, (128, 128))
            self.assertEqual(vp.dtype, np.float32)
            self.assertEqual(module.array_sha(vp), module.EVALUATION_SHA256[i])
        manifest = json.loads((ROOT / "results/redsea_sensitivity/"
                               "redsea_velocity_sweep_20261013_222242/"
                               "case_manifest.json").read_text())
        self.assertEqual(len(manifest["original_cases"]), 20)
        for case in manifest["original_cases"]:
            self.assertEqual(case["original_velocity_sha256"],
                             module.EVALUATION_SHA256[case["model_id"]])

    def test_redsea_reused_summary_rows(self):
        folder = ROOT / "results/redsea_sensitivity"
        sweep = folder / "redsea_velocity_sweep_20261013_222242/sweep_summary.csv"
        combined = folder / "redsea_genwp_comparison_20261014_194013/summary_by_alpha.csv"
        with sweep.open(newline="") as stream:
            srows = list(csv.DictReader(stream))
        with combined.open(newline="") as stream:
            crows = list(csv.DictReader(stream))
        self.assertEqual(len(srows), 16)
        self.assertEqual(len(crows), 32)
        lookup = {(r["label"], r["alpha"]): r for r in crows}
        for row in srows:
            entry = lookup[(row["label"], row["alpha"])]
            for metric in ("mae_pred100_mean", "mse_pred100_mean",
                           "relative_l2_pred100_mean"):
                self.assertEqual(row[metric], entry[metric])

    def test_final_accuracy_agrees_with_cache_benchmark(self):
        with (ROOT / "results/final66_accuracy/updated_66_per_case.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            final = list(csv.DictReader(stream))
        with (ROOT / "results/cache/accuracy_per_case.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            cache = list(csv.DictReader(stream))
        self.assertEqual(len(final), 66 * 7)
        direct = {row["case"]: float(row["mae"])
                  for row in final if row["model"] == "long100"}
        measured = {row["case"]: float(row["mae"])
                    for row in cache if row["method"] == "MCWP_standard"}
        self.assertEqual(len(direct), 66)
        self.assertEqual(set(direct), set(measured))
        for case, value in direct.items():
            self.assertAlmostEqual(value, measured[case], places=12, msg=case)

    def test_pluto_prepared_file_hashes(self):
        manifest = json.loads((ROOT / "datasets/pluto/manifest.json").read_text())
        self.assertEqual(manifest["model_count"], 3)
        self.assertEqual(manifest["shot_count"], 15)
        for model in manifest["models"]:
            path = ROOT / "datasets/pluto" / model["path"]
            self.assertEqual(sha_file(path), model["sha256"])


if __name__ == "__main__":
    unittest.main()
