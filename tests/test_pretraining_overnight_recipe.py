from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_epiplex_generic_8h100_overnight.sh"


class PretrainingOvernightRecipeTests(unittest.TestCase):
    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True, cwd=REPO_ROOT)

    def test_dry_run_emits_full_recipe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="overnight-recipe-") as tmpdir:
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "RUN_ROOT": str(Path(tmpdir) / "run"),
                    "PYTHON_BIN": "/usr/bin/python3",
                    "TOKENIZER_PYTHON_BIN": "/usr/bin/python3",
                    "TOKENIZER_REPO_ROOT": "/tmp/epiplex_tokenizer_trainer",
                }
            )
            result = subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                check=True,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
        output = result.stdout
        self.assertIn("[1/6] build final manifest", output)
        self.assertIn("build_pretraining_manifest.py", output)
        self.assertIn("--include-dclm", output)
        self.assertIn("[2/6] fit final tokenizer", output)
        self.assertIn("train_tokenizer.py", output)
        self.assertIn("--tokenizer-task generic", output)
        self.assertIn("[3/6] pack final corpus", output)
        self.assertIn("pack_pretraining_corpus.py", output)
        self.assertIn("[4/6] validate packed manifest", output)
        self.assertIn("validate_packed_pretraining_manifest.py", output)
        self.assertIn("--require-corpus dclm", output)
        self.assertIn("--require-corpus-prefix oscar", output)
        self.assertIn("[5/6] ensure Hugging Face checkpoint repo", output)
        self.assertIn("ensure_hf_checkpoint_repo.py", output)
        self.assertIn("--require-enabled", output)
        self.assertIn("[6/6] launch overnight 7B run", output)
        self.assertIn("launch_pretraining_lm_8h100_7b.sh", output)

    def test_dry_run_respects_overrides(self) -> None:
        with tempfile.TemporaryDirectory(prefix="overnight-recipe-") as tmpdir:
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "RUN_ROOT": str(Path(tmpdir) / "run"),
                    "PYTHON_BIN": "/usr/bin/python3",
                    "TOKENIZER_PYTHON_BIN": "/usr/bin/python3",
                    "TOKENIZER_REPO_ROOT": "/tmp/epiplex_tokenizer_trainer",
                    "DCLM_MAX_DOCUMENTS": "123456",
                    "TOKENIZER_SAMPLE_MAX_DOCUMENTS": "43210",
                    "TRAIN_TIMEOUT_HOURS": "12",
                    "TRAIN_STEPS": "3456",
                    "MIN_TRAIN_SEQUENCES": "222222",
                }
            )
            result = subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                check=True,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
        output = result.stdout
        self.assertIn("--dclm-max-documents 123456", output)
        self.assertIn("--sample-max-documents 43210", output)
        self.assertIn("--min-train-sequences 222222", output)
        self.assertIn("STEPS=3456", output)


if __name__ == "__main__":
    unittest.main()
