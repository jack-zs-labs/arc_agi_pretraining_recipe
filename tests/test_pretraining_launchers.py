from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_8 = REPO_ROOT / "scripts" / "launch_pretraining_lm_8h100.sh"
LAUNCH_48 = REPO_ROOT / "scripts" / "launch_pretraining_lm_48h100.sh"
LAUNCH_8_7B = REPO_ROOT / "scripts" / "launch_pretraining_lm_8h100_7b.sh"
LAUNCH_48_7B = REPO_ROOT / "scripts" / "launch_pretraining_lm_48h100_7b.sh"


class PretrainingLauncherTests(unittest.TestCase):
    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCH_8)], check=True, cwd=REPO_ROOT)
        subprocess.run(["bash", "-n", str(LAUNCH_48)], check=True, cwd=REPO_ROOT)
        subprocess.run(["bash", "-n", str(LAUNCH_8_7B)], check=True, cwd=REPO_ROOT)
        subprocess.run(["bash", "-n", str(LAUNCH_48_7B)], check=True, cwd=REPO_ROOT)

    def test_launch_8_dry_run_defaults_to_fsdp(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "PYTHON_BIN": "/usr/bin/python3",
                "PACKED_MANIFEST": "/tmp/packed_manifest.json",
                "OUTPUT_ROOT": "/tmp/pretraining_lm_8h100",
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCH_8)],
            check=True,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        output = result.stdout
        self.assertIn("--sharding-strategy fsdp_full_shard", output)
        self.assertIn("--activation-checkpointing", output)

    def test_launch_48_dry_run_respects_sharding_overrides(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "PYTHON_BIN": "/usr/bin/python3",
                "NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "PACKED_MANIFEST": "/tmp/packed_manifest.json",
                "OUTPUT_ROOT": "/tmp/pretraining_lm_48h100",
                "SHARDING_STRATEGY": "ddp",
                "ACTIVATION_CHECKPOINTING": "0",
                "PREFETCH_BATCHES": "12",
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCH_48)],
            check=True,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        output = result.stdout
        self.assertIn("--sharding-strategy ddp", output)
        self.assertIn("--no-activation-checkpointing", output)
        self.assertIn("--prefetch-batches 12", output)

    def test_launch_8_7b_dry_run_uses_large_model_defaults(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "PYTHON_BIN": "/usr/bin/python3",
                "PACKED_MANIFEST": "/tmp/packed_manifest.json",
                "OUTPUT_ROOT": "/tmp/pretraining_lm_8h100_7b",
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCH_8_7B)],
            check=True,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        output = result.stdout
        self.assertIn("--sharding-strategy fsdp_full_shard", output)
        self.assertIn("--activation-checkpointing", output)
        self.assertIn("--batch-size 1", output)
        self.assertIn("--hidden-size 4096", output)
        self.assertIn("--num-layers 32", output)

    def test_launch_8_dry_run_sources_repo_local_env_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as handle:
            handle.write("LEARNING_RATE=1.234e-4\n")
            env_path = handle.name
        self.addCleanup(lambda: os.path.exists(env_path) and os.unlink(env_path))

        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "PYTHON_BIN": "/usr/bin/python3",
                "PACKED_MANIFEST": "/tmp/packed_manifest.json",
                "OUTPUT_ROOT": "/tmp/pretraining_lm_8h100",
                "PRETRAINING_ENV_FILE": env_path,
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCH_8)],
            check=True,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertIn("--learning-rate 1.234e-4", result.stdout)


if __name__ == "__main__":
    unittest.main()
