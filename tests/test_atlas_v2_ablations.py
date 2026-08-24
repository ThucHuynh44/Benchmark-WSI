import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.atlas_v2_registry import MECHANISM_FIELDS, PAIRWISE, SETTING_IDS, load_registry
from scripts.run_atlas_v2_ablations import PILOT_VARIANTS, build_command, main as run_main
from scripts.summarize_atlas_v2_ablations import METRICS, markdown, main as summary_main


ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "configs/atlas_v2_ablations.yaml"


class RegistryTests(unittest.TestCase):
    def test_exactly_eight_unique_settings_and_hashes(self):
        registry = load_registry(REGISTRY)
        self.assertEqual(tuple(registry["variants"]), SETTING_IDS)
        hashes = [entry["config_hash"] for entry in registry["variants"].values()]
        self.assertEqual(len(hashes), 8)
        self.assertEqual(len(set(hashes)), 8)
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
        }
        for variant_id, values in expected.items():
            overrides = variants[variant_id]["overrides"]
            actual = tuple(overrides[field] for field in MECHANISM_FIELDS) + (overrides["buffer_size"],)
            self.assertEqual(actual, values)

    def test_commands_use_separate_result_tree_and_full_bag_flags(self):
        registry = load_registry(REGISTRY)
        for variant in registry["variants"].values():
            command = build_command(registry, variant, 2)
            self.assertIn("atlas_v2", command)
            self.assertIn("feather", command)
            self.assertIn(f"ablations/atlas_v2/{variant['id']}/fold_2", command)
            self.assertNotIn("pmp_k", " ".join(command))


class RunnerTests(unittest.TestCase):
    def test_pilot_dry_run_is_exactly_six_by_three(self):
        argv = ["dry-run", "--variants", *PILOT_VARIANTS, "--folds", "0,1,2", "--gpus", "0"]
        stream = io.StringIO()
        with patch("scripts.run_atlas_v2_ablations.inspect_run", return_value="missing"), contextlib.redirect_stdout(stream):
            code = run_main(argv)
        output = stream.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("commands=18", output)
        self.assertEqual(output.count("variant="), 18)
        for field in ("LoRA=", "Replay=", "buffer=", "full_bag=", "Prototype=", "Realignment=", "Prompt=", "NCE="):
            self.assertEqual(output.count(field), 18)
        self.assertNotIn("atlasv2_lora_replay_proto_realign_prompt fold=", output)


class SummaryTests(unittest.TestCase):
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
        ladder = rendered.split("# ATLAS-v2 Additive Ladder", 1)[1].split("# ATLAS-v2 Frozen", 1)[0]
        positions = [ladder.index(f"| {variant_id} |") for variant_id in SETTING_IDS[:5]]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("No winner is selected automatically", rendered)

    def test_summary_smoke_writes_only_to_requested_tmp_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.summarize_atlas_v2_ablations.inspect_run", return_value="missing"
        ):
            code = summary_main(["--output", directory])
            self.assertEqual(code, 0)
            output = Path(directory)
            self.assertTrue((output / "atlas_v2_per_fold.csv").is_file())
            self.assertTrue((output / "atlas_v2_summary.csv").is_file())
            self.assertTrue((output / "atlas_v2_tables.md").is_file())


if __name__ == "__main__":
    unittest.main()
