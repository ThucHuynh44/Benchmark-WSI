import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.atlas_ablation_registry import load_registry
from scripts.run_atlas_ablations import build_command, experiment_desc, inspect_run
from scripts.summarize_atlas_ablations import summarize_folds, summarize_mechanisms
from configs.experiment_loader import config_to_argv


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/atlas_mil_ablations.yaml"


class AtlasAblationRegistryTests(unittest.TestCase):
    def test_yaml_boolean_optional_flags_preserve_false(self):
        self.assertEqual(config_to_argv({"atlas_replay": False}), ["--no-atlas_replay"])
        self.assertEqual(config_to_argv({"atlas_diagnostics": True}), ["--atlas_diagnostics"])

    def test_registry_resolves_exact_unique_matrix(self):
        registry = load_registry(REGISTRY)
        variants = registry["variants"]
        self.assertEqual(len(variants), 33)
        self.assertEqual(variants["sgd_ft"]["group"], "external_reference")
        self.assertNotEqual(variants["sgd_ft"]["group"], "additive")
        self.assertFalse(variants["atlas_ce_noreplay"]["overrides"]["atlas_replay"])
        self.assertEqual(
            variants["centroid_with_prompt_fallback"]["overrides"]["atlas_prompt_weight"],
            0.0,
        )
        hashes = {entry["config_hash"] for entry in variants.values()}
        self.assertEqual(len(hashes), 33)
        prompt_members = registry["axis_members"]["atlas_prompt_weight"]
        self.assertEqual(prompt_members[0]["variant_id"], "centroid_with_prompt_fallback")
        self.assertEqual(prompt_members[2]["variant_id"], "full")
        self.assertEqual(prompt_members[-1]["variant_id"], "prompt_only")

    def test_commands_are_fold_isolated_and_boolean_safe(self):
        registry = load_registry(REGISTRY)
        variant = registry["variants"]["atlas_ce_noreplay"]
        command = build_command(registry, variant, 3)
        self.assertIn("--no-atlas_replay", command)
        self.assertIn("ablations/atlas_mil/atlas_ce_noreplay/fold_3", command)
        self.assertEqual(command[command.index("--folds") + 1], "3")

    def test_resume_audit_detects_complete_and_hash_mismatch(self):
        registry = load_registry(REGISTRY)
        variant = registry["variants"]["full"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "results" / experiment_desc(registry, "full", 0)
            for mode in ("class_il", "task_il"):
                target = run_dir / "evaluation" / mode
                target.mkdir(parents=True, exist_ok=True)
                with (target / "eval_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["fold", "after_task", "eval_task"])
                    writer.writeheader()
                    for after in range(10):
                        for evaluated in range(after + 1):
                            writer.writerow({"fold": 0, "after_task": after, "eval_task": evaluated})
            manifest = {
                "ablation_id": "full",
                "ablation_config_hash": variant["config_hash"],
                "folds": [0],
            }
            manifest_path = run_dir / "evaluation/class_il/run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("scripts.run_atlas_ablations.REPO_ROOT", root):
                self.assertEqual(inspect_run(registry, variant, 0), "complete")
                with (run_dir / "evaluation/task_il/eval_matrix.csv").open(
                    "a", newline="", encoding="utf-8"
                ) as handle:
                    csv.writer(handle).writerow([0, 0, 0])
                self.assertEqual(inspect_run(registry, variant, 0), "incomplete")
                manifest["ablation_config_hash"] = "wrong"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(inspect_run(registry, variant, 0), "mismatch")


class AtlasAblationSummaryTests(unittest.TestCase):
    def test_paired_delta_uses_matching_folds(self):
        registry = load_registry(REGISTRY)
        rows = []
        for variant, values in (("full", [0.5, 0.7]), ("wo_nce", [0.4, 0.8])):
            entry = registry["variants"][variant]
            for fold, value in enumerate(values):
                row = {
                    "variant_id": variant, "group": entry["group"],
                    "label": entry["label"], "factor": entry["factor"],
                    "value": entry["value"], "model": entry["model"],
                    "fold": fold, "status": "complete",
                }
                for metric in (
                    "mACC", "bACC", "masked_bACC", "BWT", "FGT", "auroc",
                    "training_time", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib",
                    "total_parameters", "trainable_parameters", "parameter_growth",
                ):
                    row[metric] = value
                rows.append(row)
        summaries = {row["variant_id"]: row for row in summarize_folds(registry, rows)}
        self.assertAlmostEqual(summaries["wo_nce"]["mACC_delta_vs_full_mean"], 0.0)
        self.assertAlmostEqual(summaries["wo_nce"]["mACC_delta_vs_full_std"], 0.1)

    def test_mechanism_overall_is_macro_over_task_per_fold(self):
        rows = [
            {"variant_id": "full", "fold": 0, "task": 0, "semantic_rho": 0.0},
            {"variant_id": "full", "fold": 0, "task": 1, "semantic_rho": 1.0},
            {"variant_id": "full", "fold": 1, "task": 0, "semantic_rho": 0.5},
        ]
        summary = summarize_mechanisms(rows)
        overall = next(row for row in summary if row["task"] == "overall")
        self.assertAlmostEqual(overall["semantic_rho_mean"], 0.5)
        self.assertEqual(overall["fold_count"], 2)


if __name__ == "__main__":
    unittest.main()
