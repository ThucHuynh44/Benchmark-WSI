import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.atlas_ablation_registry import load_registry
from scripts.run_atlas_ablations import build_command, experiment_desc, inspect_run
from scripts.summarize_atlas_ablations import (
    _markdown,
    summarize_folds,
    summarize_mechanisms,
)
from configs.experiment_loader import config_to_argv


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/atlas_mil_ablations.yaml"


class AtlasAblationRegistryTests(unittest.TestCase):
    PRESERVED_HASHES = {
        "full": "fca85b9fa283f85567c42726375f4ff2c88213dea80814e6eb480368635ac4ec",
        "atlas_ce": "d942c2866a1696cae98bee6ae12dcbe93837dc41b2fba0f9b599ef5e7d10d689",
        "add_attention": "8678cd391fea1e5811addf6f0230ebfb3e0721db7947f73c7b7524a2ef03637c",
        "add_nce": "5b5f98bb1d69490b6e18c5ffe3ca6369e9db34127fb76063a1d9b4cbe803f659",
        "add_reconstruction": "ea7167607792515e07724b0d1a1f50d0b1a47d8330382ffc44fb6af7ad064b1c",
        "solm_none": "b3404d5891b06ca14fc35b2685355eee5d55b49fdffb7eb7f0bf3e91dfef71ce",
        "solm_hard": "6b2411cc9d5746216ffd0959c2bcfda17d29dea54184e1bad354cdbcb04783d2",
        "wo_replay": "5d5cc02361fffd64b2259cfe8de463ef201725460ea66688ddc4bd92a69641d2",
        "wo_nce": "5426758d9ab8640d9780ac205e1cbdf4ebec23ce07edbae2c14c3bccf7fecc89",
        "wo_attention": "775f4aa00cd59be6097630ec2ab5ebba7eba121a574303528458cca0e177305a",
        "prompt_only": "62a6c415b1b67de50e7a855df4beaa98876eb9799cc4ff75ab4d2f3ae55852cd",
        "centroid_with_prompt_fallback": "1beaef972b19c9df2103abdebb6dfe6ab06496694ae8c45cd0be7958579dbfef",
    }

    def test_yaml_boolean_optional_flags_preserve_false(self):
        self.assertEqual(config_to_argv({"atlas_replay": False}), ["--no-atlas_replay"])
        self.assertEqual(config_to_argv({"atlas_diagnostics": True}), ["--atlas_diagnostics"])
        self.assertEqual(
            config_to_argv({"atlas_lora_enabled": False}),
            ["--no-atlas_lora_enabled"],
        )

    def test_registry_resolves_exact_unique_matrix(self):
        registry = load_registry(REGISTRY)
        variants = registry["variants"]
        self.assertEqual(len(variants), 34)
        self.assertEqual(variants["sgd_ft"]["group"], "external_reference")
        self.assertNotEqual(variants["sgd_ft"]["group"], "additive")
        self.assertFalse(variants["atlas_ce_noreplay"]["overrides"]["atlas_replay"])
        self.assertEqual(
            variants["centroid_with_prompt_fallback"]["overrides"]["atlas_prompt_weight"],
            0.0,
        )
        hashes = {entry["config_hash"] for entry in variants.values()}
        self.assertEqual(len(hashes), 34)
        self.assertFalse(variants["full_no_lora"]["overrides"]["atlas_lora_enabled"])
        self.assertNotIn("atlas_lora_enabled", variants["full"]["overrides"])
        for variant_id, expected_hash in self.PRESERVED_HASHES.items():
            self.assertEqual(variants[variant_id]["config_hash"], expected_hash)
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
        no_lora = build_command(registry, registry["variants"]["full_no_lora"], 0)
        self.assertIn("--no-atlas_lora_enabled", no_lora)

    def test_attention_pilot_changes_only_attention_weight(self):
        registry = load_registry(REGISTRY)
        variants = registry["variants"]
        expected = {
            "atlas_ce": 0.0,
            "att_w025": 0.25,
            "att_w05": 0.5,
            "add_attention": 1.0,
        }
        normalized = {}
        for variant_id, weight in expected.items():
            overrides = dict(variants[variant_id]["overrides"])
            self.assertEqual(overrides["attention_weight"], weight)
            self.assertEqual(overrides["atlas_nce_weight"], 0.0)
            self.assertEqual(overrides["reconstruction_weight"], 0.0)
            self.assertEqual(overrides["manifold_weight"], 0.0)
            overrides.pop("attention_weight")
            normalized[variant_id] = overrides
        self.assertTrue(all(
            value == normalized["atlas_ce"] for value in normalized.values()
        ))

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

    def test_markdown_contains_architecture_and_ordered_attention_pilot(self):
        registry = load_registry(REGISTRY)
        rows = []
        for variant in registry["variants"].values():
            row = {"variant_id": variant["id"], "completed_folds": 0}
            for metric in ("mACC", "bACC", "masked_bACC", "BWT", "FGT"):
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
            rows.append(row)
        rendered = _markdown(registry, rows)
        self.assertIn("| full_no_lora |", rendered)
        pilot = rendered.split("## Attention-weight pilot", 1)[1].split("## ", 1)[0]
        positions = [
            pilot.index(f"| {variant_id} |")
            for variant_id in ("atlas_ce", "att_w025", "att_w05", "add_attention")
        ]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
