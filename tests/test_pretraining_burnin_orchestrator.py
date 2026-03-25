from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_epiplex_generic_8h100_burnin.sh"


class PretrainingBurninOrchestratorTests(unittest.TestCase):
    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True, cwd=REPO_ROOT)

    def test_dry_run_emits_all_expected_steps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="burnin-orchestrator-") as tmpdir:
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
        self.assertIn("[1/5] build manifest", output)
        self.assertIn("build_pretraining_manifest.py", output)
        self.assertIn("[2/5] fit tokenizer", output)
        self.assertIn("train_tokenizer.py", output)
        self.assertIn("--tokenizer-task generic", output)
        self.assertIn("[3/5] pack corpus", output)
        self.assertIn("pack_pretraining_corpus.py", output)
        self.assertIn("[4/5] 8xH100 burn-in", output)
        self.assertIn("launch_pretraining_lm_8h100.sh", output)
        self.assertIn("burnin_8xh100_step50", output)
        self.assertIn("SHARDING_STRATEGY=fsdp_full_shard", output)
        self.assertIn("ACTIVATION_CHECKPOINTING=1", output)
        self.assertIn("[5/5] resume check", output)
        self.assertIn("burnin_8xh100_resume75", output)
        self.assertIn("RESUME_FROM=", output)

    def test_dry_run_respects_overrides(self) -> None:
        with tempfile.TemporaryDirectory(prefix="burnin-orchestrator-") as tmpdir:
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "RUN_ROOT": str(Path(tmpdir) / "run"),
                    "PYTHON_BIN": "/usr/bin/python3",
                    "TOKENIZER_PYTHON_BIN": "/usr/bin/python3",
                    "TOKENIZER_REPO_ROOT": "/tmp/epiplex_tokenizer_trainer",
                    "TOKENIZER_TASK": "reasoning_graph",
                    "DCLM_MAX_DOCUMENTS": "1234",
                    "TOKENIZER_SAMPLE_MAX_DOCUMENTS": "4321",
                    "SHARDING_STRATEGY": "fsdp_shard_grad_op",
                    "ACTIVATION_CHECKPOINTING": "0",
                    "STEPS": "17",
                    "RESUME_STEPS": "23",
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
        self.assertIn("--dclm-max-documents 1234", output)
        self.assertIn("--sample-max-documents 4321", output)
        self.assertIn("--tokenizer-task reasoning_graph", output)
        self.assertIn("SHARDING_STRATEGY=fsdp_shard_grad_op", output)
        self.assertIn("ACTIVATION_CHECKPOINTING=0", output)
        self.assertIn("STEPS=17", output)
        self.assertIn("STEPS=23", output)


if __name__ == "__main__":
    unittest.main()
