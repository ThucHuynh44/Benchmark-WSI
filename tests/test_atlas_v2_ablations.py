import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.atlas_v2_registry import (
    COMEL_SETTING_ID, FROZEN_PROTO_LDA_ID, MECHANISM_FIELDS, PAIRWISE,
    PROTO_FACTORIAL_IDS, SETTING_IDS, load_registry,
)
from scripts.run_atlas_v2_ablations import (
    PILOT_VARIANTS, build_command, main as run_main,
    valid_calibration_manifest,
)
from scripts.summarize_atlas_v2_ablations import (
    METRICS, _memory_from_manifest, markdown, main as summary_main,
)


ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "configs/atlas_v2_ablations.yaml"


class RegistryTests(unittest.TestCase):
    def test_legacy_resolved_hashes_are_frozen(self):
        expected = {
            "atlasv2_base_frozen": "445ef223888faa607618a60c11c200937cac39f61b62661f00e86cee900c1bbe",
            "atlasv2_base_lora": "e6b752a9ca303314642f3f6cced57fc8372c4937e157835f868aa6f3de1bb6a7",
            "atlasv2_lora_replay": "da399fb662ef7d4aeb24b6a9aa596c83c67b12417c7c8d57891f2a3c3e6005a7",
            "atlasv2_lora_replay_proto": "6b2f02385f1643ce8f82903e5b17f8dbe182941f83598b5e10c8ef64372780fc",
            "atlasv2_lora_replay_proto_realign": "a6bd713dc68bd465006f945b5db459259c2761a4094543ab4e258b475ab9417e",
            "atlasv2_frozen_proto": "ee448c0d65c42c3cca07b35837f9042909695a6f50b09af5bf69f328df46fc73",
            "atlasv2_lora_replay_proto_realign_prompt": "c088520db235ef6d78d69153b84f1009a435dd91af29265ba1fffd327274e76a",
            "atlasv2_lora_replay_proto_realign_prompt_nce": "224e1f65b3c5d943bdcd14bcb482c15819f42ff7f1c993491bbd7a48951f9308",
            "atlasv2_base_lora_svd_orthogonal": "f225bd7f8c3c24433816678877403b1bf7370f5733e47ca74601fd88a7af504a",
            "atlasv2_replay_proto": "61d30c701e9db3d73a4e85579699591b3990fa69fb4ef10bd840908a53e1f0c0",
            "atlasv2_lora_proto": "ba4b1fa9297e21a1fbcb7386933a152f8dd5fd4528c7e95a302b3ed251bfc077",
            "atlasv2_base_lora_comel_owlora": "1088dfb6537680b17d2e4703c91ec952c506a61a218e9347f15151e37ac1f1fa",
            "atlasv2_frozen_proto_oas_lda": "2f78f7e375a3bbb148555ec938d41574618074ba07cf14b5322a29a3b7ad559b",
        }
        variants = load_registry(REGISTRY)["variants"]
        self.assertEqual(
            {variant_id: variants[variant_id]["config_hash"] for variant_id in expected},
            expected,
        )

    def test_all_unique_settings_and_hashes(self):
        registry = load_registry(REGISTRY)
        self.assertEqual(tuple(registry["variants"]), SETTING_IDS)
        hashes = [entry["config_hash"] for entry in registry["variants"].values()]
        self.assertEqual(len(hashes), len(SETTING_IDS))
        self.assertEqual(len(set(hashes)), len(SETTING_IDS))
        for variant in registry["variants"].values():
            overrides = variant["overrides"]
            self.assertTrue(all(field in overrides for field in MECHANISM_FIELDS))
            self.assertEqual(
                overrides["buffer_size"], 30 if overrides["atlasv2_replay"] else 0
            )

    def test_pairwise_scientific_differences_are_isolated(self):
        variants = load_registry(REGISTRY)["variants"]
        for left_id, right_id, mechanism in PAIRWISE:
            left = variants[left_id]["overrides"]
            right = variants[right_id]["overrides"]
            differences = {
                key for key in set(left) | set(right) if left.get(key) != right.get(key)
            }
            expected = {mechanism, "buffer_size"} if mechanism == "atlasv2_replay" else {mechanism}
            self.assertEqual(differences, expected)

    def test_resolved_configuration_matches_final_table(self):
        variants = load_registry(REGISTRY)["variants"]
        expected = {
            "atlasv2_base_frozen": (False, False, False, False, False, False, 0),
            "atlasv2_base_lora": (True, False, False, False, False, False, 0),
            "atlasv2_lora_replay": (True, True, False, False, False, False, 30),
            "atlasv2_lora_replay_proto": (True, True, True, False, False, False, 30),
            "atlasv2_lora_replay_proto_realign": (True, True, True, True, False, False, 30),
            "atlasv2_frozen_proto": (False, False, True, False, False, False, 0),
            "atlasv2_lora_replay_proto_realign_prompt": (True, True, True, True, True, False, 30),
            "atlasv2_lora_replay_proto_realign_prompt_nce": (True, True, True, True, True, True, 30),
            "atlasv2_base_lora_svd_orthogonal": (True, False, False, False, False, False, 0),
            "atlasv2_replay_proto": (False, True, True, False, False, False, 30),
            "atlasv2_lora_proto": (True, False, True, False, False, False, 0),
            "atlasv2_base_lora_comel_owlora": (True, False, False, False, False, False, 0),
            "atlasv2_frozen_proto_oas_lda": (False, False, True, False, False, False, 0),
        }
        for variant_id, values in expected.items():
            overrides = variants[variant_id]["overrides"]
            actual = tuple(overrides[field] for field in MECHANISM_FIELDS) + (overrides["buffer_size"],)
            self.assertEqual(actual, values)
        geometry = variants["atlasv2_base_lora_svd_orthogonal"]["overrides"]
        self.assertTrue(geometry["atlasv2_svd_orthogonal"])
        self.assertEqual(geometry["atlasv2_svd_energy"], 0.99)
        comel = variants[COMEL_SETTING_ID]["overrides"]
        self.assertTrue(comel["atlasv2_comel_owlora"])
        self.assertEqual(comel["atlasv2_comel_svd_energy"], 0.99)
        self.assertEqual(comel["atlasv2_comel_orthogonal_weight"], 1.0)
        lda = variants[FROZEN_PROTO_LDA_ID]["overrides"]
        self.assertTrue(lda["atlasv2_prototype_lda"])
        self.assertFalse(lda["atlasv2_train_classifier"])

    def test_prototype_replay_lora_factorial_has_all_four_cells(self):
        variants = load_registry(REGISTRY)["variants"]
        expected = {
            "atlasv2_frozen_proto": (False, False, False),
            "atlasv2_replay_proto": (False, True, False),
            "atlasv2_lora_proto": (True, False, True),
            "atlasv2_lora_replay_proto": (True, True, True),
        }
        self.assertEqual(set(PROTO_FACTORIAL_IDS), set(expected))
        for variant_id, cell in expected.items():
            overrides = variants[variant_id]["overrides"]
            self.assertEqual(
                (
                    overrides["atlasv2_lora"], overrides["atlasv2_replay"],
                    overrides["atlasv2_train_classifier"],
                ),
                cell,
            )
            self.assertTrue(overrides["atlasv2_prototype"])
            self.assertFalse(overrides["atlasv2_realign"])
            self.assertFalse(overrides["atlasv2_prompt"])
            self.assertFalse(overrides["atlasv2_nce"])

    def test_commands_use_separate_result_tree_and_full_bag_flags(self):
        registry = load_registry(REGISTRY)
        for variant in registry["variants"].values():
            command = build_command(registry, variant, 2)
            self.assertIn("atlas_v2", command)
            self.assertIn("feather", command)
            self.assertIn(f"ablations/atlas_v2/{variant['id']}/fold_2", command)
            self.assertNotIn("pmp_k", " ".join(command))


class RunnerTests(unittest.TestCase):
    def test_distribution_completion_requires_validation_only_calibration_manifest(self):
        variant = load_registry(REGISTRY)["variants"]["atlasv2_frozen_atlas_tf"]
        payload = {
            "distribution_state_version": 1,
            "fold": 0,
            "runtime_slide_embedding_dim": 512,
            "contains_train_embeddings": False,
            "contains_validation_embeddings": False,
            "contains_test_embeddings": False,
            "history": [
                {"split": "validation", "contains_test_cache": False}
                for _ in range(10)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold_0.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(valid_calibration_manifest(path, variant, 0, 10))
            payload["history"][0]["contains_test_cache"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(valid_calibration_manifest(path, variant, 0, 10))

    def test_pilot_dry_run_is_exactly_six_by_three(self):
        argv = ["dry-run", "--variants", *PILOT_VARIANTS, "--folds", "0,1,2", "--gpus", "0"]
        stream = io.StringIO()
        with patch("scripts.run_atlas_v2_ablations.inspect_run", return_value="missing"), contextlib.redirect_stdout(stream):
            code = run_main(argv)
        output = stream.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("commands=18", output)
        self.assertEqual(output.count("variant="), 18)
        for field in (
            "LoRA=", "Replay=", "buffer=", "full_bag=", "Prototype=",
            "Realignment=", "Prompt=", "NCE=", "LinearCE=", "CoMEL=",
            "TrainLDA=",
        ):
            self.assertEqual(output.count(field), 18)
        self.assertNotIn("atlasv2_lora_replay_proto_realign_prompt fold=", output)


class SummaryTests(unittest.TestCase):
    def test_memory_accounting_uses_manifest_without_checkpoint_loading(self):
        variants = load_registry(REGISTRY)["variants"]
        replay = variants["atlasv2_lora_replay"]
        accounting = {
            "retained_wsis": 30,
            "retained_patch_rows": 81234,
            "replay_memory_mib": 239.5,
        }
        self.assertEqual(
            _memory_from_manifest({"replay_memory_accounting": accounting}, replay),
            accounting,
        )
        missing = _memory_from_manifest({}, replay)
        self.assertTrue(all(math.isnan(missing[field]) for field in missing))
        no_replay = variants["atlasv2_frozen_proto"]
        self.assertEqual(
            _memory_from_manifest({}, no_replay),
            {"retained_wsis": 0, "retained_patch_rows": 0, "replay_memory_mib": 0.0},
        )

    def _empty_rows(self):
        rows = []
        for variant_id in SETTING_IDS:
            row = {"variant_id": variant_id, "completed_folds": 0}
            for metric in METRICS:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
            rows.append(row)
        return rows

    def test_markdown_has_required_ordered_sections_without_winner(self):
        rendered = markdown(self._empty_rows())
        self.assertIn("# ATLAS-v2 Additive Ladder", rendered)
        self.assertIn("# ATLAS-v2 Frozen Baseline", rendered)
        self.assertIn("# ATLAS-v2 Semantic Extensions", rendered)
        self.assertIn("# ATLAS-v2 LoRA Geometry Extension", rendered)
        self.assertIn("# ATLAS-v2 CoMEL LoRA Strategy", rendered)
        self.assertIn("# ATLAS-v2 Frozen Prototype Extension", rendered)
        self.assertIn("# ATLAS-v2 Prototype LoRA × Replay Factorial", rendered)
        factorial = rendered.split(
            "# ATLAS-v2 Prototype LoRA × Replay Factorial", 1
        )[1]
        positions = [factorial.index(f"| {variant_id} |") for variant_id in PROTO_FACTORIAL_IDS]
        self.assertEqual(positions, sorted(positions))
        ladder = rendered.split("# ATLAS-v2 Additive Ladder", 1)[1].split("# ATLAS-v2 Frozen", 1)[0]
        positions = [ladder.index(f"| {variant_id} |") for variant_id in SETTING_IDS[:5]]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("No winner is selected automatically", rendered)

    def test_summary_smoke_writes_only_to_requested_tmp_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.summarize_atlas_v2_ablations._load_run_once",
            return_value=("missing", None, None, None),
        ):
            code = summary_main(["--output", directory])
            self.assertEqual(code, 0)
            output = Path(directory)
            self.assertTrue((output / "atlas_v2_per_fold.csv").is_file())
            self.assertTrue((output / "atlas_v2_summary.csv").is_file())
            self.assertTrue((output / "atlas_v2_tables.md").is_file())
            self.assertTrue((output / "replay_memory_cache.json").is_file())


if __name__ == "__main__":
    unittest.main()
