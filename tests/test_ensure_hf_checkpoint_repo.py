from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ensure_hf_checkpoint_repo.py"


class EnsureHfCheckpointRepoTests(unittest.TestCase):
    def test_dry_run_uses_env_configuration(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "HF_UPLOAD_MODE": "best_effort",
                "HF_UPLOAD_REPO_ID": "jack-zs-labs/test-nightly",
                "HF_UPLOAD_REPO_TYPE": "model",
                "HF_UPLOAD_PATH_PREFIX": "runs/nightly",
                "HF_UPLOAD_PRIVATE": "1",
                "HF_TOKEN": "hf_dummy_token",
            }
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--dry-run"],
            check=True,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "DRY_RUN")
        self.assertEqual(payload["hf_upload"]["repo_id"], "jack-zs-labs/test-nightly")
        self.assertEqual(payload["hf_upload"]["path_prefix"], "runs/nightly")
        self.assertTrue(payload["hf_upload"]["private"])

    def test_require_enabled_fails_when_not_configured(self) -> None:
        env = os.environ.copy()
        env.pop("HF_UPLOAD_MODE", None)
        env.pop("HF_UPLOAD_REPO_ID", None)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--require-enabled"],
            check=False,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HF_UPLOAD_MODE", result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
