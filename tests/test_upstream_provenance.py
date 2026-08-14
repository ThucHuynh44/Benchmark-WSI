import ast
import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "upstream"
GENERATED_FILES = {"SOURCE_MANIFEST.json", "INTERNAL_RESEARCH_ONLY.md"}
EXPECTED_SOURCES = {
    "comel_owlora": {
        "project": "CoMEL-OWLoRA",
        "repository": "https://github.com/Hyun1A/CoMEL.git",
        "commit": "9fd667994eb57e3960e36970a9509a8217d84a22",
        "core_entrypoint": "continual/main/continual_bag/cdatmil_ppl_owlora_trainer.py",
    },
    "lwsr": {
        "project": "LWSR",
        "repository": "https://github.com/OliverZXY/LWSR.git",
        "commit": "7620ef944d7dabbb20504744fd244633fc3841d1",
        "core_entrypoint": "models/lwsr.py",
    },
    "micil": {
        "project": "MICIL",
        "repository": "https://github.com/cvblab/MICIL.git",
        "commit": "7c27d197ca522a3cfe3b0629152a07858f707bdf",
        "core_entrypoint": "code_py/MICIL_train.py",
    },
    "qpmil_vl": {
        "project": "QPMIL-VL",
        "repository": "https://github.com/can-can-ya/QPMIL-VL.git",
        "commit": "3a7a7698582dec866d43eb748f8c3599f7be4391",
        "core_entrypoint": "models/model_il.py",
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class UpstreamProvenanceTests(unittest.TestCase):
    def test_manifests_cover_every_frozen_file_and_match_checksums(self):
        self.assertEqual(
            {path.name for path in UPSTREAM_ROOT.iterdir() if path.is_dir()},
            set(EXPECTED_SOURCES),
        )

        for directory, expected_source in EXPECTED_SOURCES.items():
            with self.subTest(project=directory):
                snapshot = UPSTREAM_ROOT / directory
                manifest_path = snapshot / "SOURCE_MANIFEST.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

                self.assertEqual(manifest["schema_version"], 1)
                self.assertEqual(manifest["snapshot_policy"], "byte-identical")
                for key, value in expected_source.items():
                    self.assertEqual(manifest[key], value)

                entries = manifest["files"]
                listed_paths = [entry["path"] for entry in entries]
                self.assertEqual(len(listed_paths), len(set(listed_paths)))
                self.assertEqual(listed_paths, sorted(listed_paths))

                actual_paths = sorted(
                    path.relative_to(snapshot).as_posix()
                    for path in snapshot.rglob("*")
                    if path.is_file() and path.name not in GENERATED_FILES
                )
                self.assertEqual(listed_paths, actual_paths)
                self.assertIn(manifest["core_entrypoint"], listed_paths)

                for entry in entries:
                    frozen_file = snapshot / entry["path"]
                    self.assertTrue(frozen_file.is_file())
                    self.assertEqual(_sha256(frozen_file), entry["sha256"])

    def test_restricted_snapshots_are_marked_internal_research_only(self):
        for directory in ("comel_owlora", "micil", "qpmil_vl"):
            with self.subTest(project=directory):
                marker = UPSTREAM_ROOT / directory / "INTERNAL_RESEARCH_ONLY.md"
                self.assertIn("INTERNAL_RESEARCH_ONLY", marker.read_text(encoding="utf-8"))

    def test_runtime_python_does_not_import_frozen_snapshots(self):
        violations = []
        for source in PROJECT_ROOT.rglob("*.py"):
            relative = source.relative_to(PROJECT_ROOT)
            if "third_party" in relative.parts or "tests" in relative.parts:
                continue

            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
            for node in ast.walk(tree):
                imported_name = None
                if isinstance(node, ast.Import):
                    imported_name = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported_name = [node.module or ""]

                if imported_name and any(
                    name == "third_party" or name.startswith("third_party.")
                    for name in imported_name
                ):
                    violations.append(f"{relative}:{node.lineno}")

                if not isinstance(node, ast.Call) or not node.args:
                    continue
                first_arg = node.args[0]
                if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
                    continue
                if "third_party" not in first_arg.value:
                    continue

                function_name = ""
                if isinstance(node.func, ast.Name):
                    function_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    function_name = node.func.attr
                if function_name in {"__import__", "import_module", "spec_from_file_location"}:
                    violations.append(f"{relative}:{node.lineno}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
