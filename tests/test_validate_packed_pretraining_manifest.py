from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_packed_pretraining_manifest.py"


def write_manifest(path: Path, *, include_dclm: bool = True) -> None:
    corpus_counts = {
        "oscar_scope": 1200,
        "oscar_scope_reasoning": 2400,
    }
    if include_dclm:
        corpus_counts["dclm"] = 500000
    payload = {
        "seq_len": 2048,
        "tokenizer": {
            "kind": "epiplex",
            "task": "generic",
            "vocab_size": 8192,
        },
        "splits": {
            "train": {
                "document_count": 1000,
                "sequence_count": 250000,
            }
        },
        "document_manifest_corpus_counts": {
            "train": corpus_counts,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class ValidatePackedPretrainingManifestTests(unittest.TestCase):
    def test_validation_succeeds_for_expected_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="validate-packed-") as tmpdir:
            manifest_path = Path(tmpdir) / "packed_manifest.json"
            write_manifest(manifest_path, include_dclm=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--packed-manifest",
                    str(manifest_path),
                    "--require-corpus",
                    "dclm",
                    "--require-corpus-prefix",
                    "oscar",
                ],
                check=True,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )
        self.assertIn('"status": "READY"', result.stdout)

    def test_validation_rejects_missing_required_corpus(self) -> None:
        with tempfile.TemporaryDirectory(prefix="validate-packed-") as tmpdir:
            manifest_path = Path(tmpdir) / "packed_manifest.json"
            write_manifest(manifest_path, include_dclm=False)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--packed-manifest",
                    str(manifest_path),
                    "--require-corpus",
                    "dclm",
                    "--require-corpus-prefix",
                    "oscar",
                ],
                check=False,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required train corpora", result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
