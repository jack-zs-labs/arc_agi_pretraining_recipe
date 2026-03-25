from __future__ import annotations

import argparse
import json
import os
import sys


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and create the Hugging Face repo used for checkpoint uploads.")
    parser.add_argument("--mode", choices=("disabled", "best_effort", "required"), default="")
    parser.add_argument("--repo-id", type=str, default="")
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"), default="")
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--path-prefix", type=str, default="")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, object] | None:
    mode = str(args.mode or os.environ.get("HF_UPLOAD_MODE", "")).strip().lower() or "disabled"
    repo_id = str(args.repo_id or os.environ.get("HF_UPLOAD_REPO_ID", "")).strip()
    if mode == "disabled" or not repo_id:
        return None
    repo_type = str(args.repo_type or os.environ.get("HF_UPLOAD_REPO_TYPE", "model")).strip() or "model"
    token = str(
        args.token
        or os.environ.get("HF_UPLOAD_TOKEN", "")
        or os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    ).strip()
    private = args.private
    if private is None:
        private = _env_bool("HF_UPLOAD_PRIVATE", False)
    path_prefix = str(args.path_prefix or os.environ.get("HF_UPLOAD_PATH_PREFIX", "")).strip().strip("/")
    return {
        "mode": mode,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "token": token,
        "private": bool(private),
        "path_prefix": path_prefix,
    }


def require_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "Hugging Face checkpoint upload readiness was requested, but `huggingface_hub` is not installed. "
            "Install requirements.txt before launching the paid run."
        ) from exc
    return HfApi


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    if config is None:
        if args.require_enabled:
            raise SystemExit(
                "Hugging Face checkpoint upload is not configured. Set HF_UPLOAD_MODE and HF_UPLOAD_REPO_ID "
                "before launching the overnight run."
            )
        print(json.dumps({"status": "DISABLED", "hf_upload": None}, indent=2))
        return

    token = str(config["token"]).strip()
    if not token:
        raise SystemExit(
            "HF upload is enabled but no token was found. Set HF_TOKEN, HF_UPLOAD_TOKEN, or --token before launch."
        )

    payload = {
        "status": "READY" if not args.dry_run else "DRY_RUN",
        "hf_upload": {
            "mode": config["mode"],
            "repo_id": config["repo_id"],
            "repo_type": config["repo_type"],
            "private": bool(config["private"]),
            "path_prefix": config["path_prefix"],
            "repo_url": f"https://huggingface.co/{config['repo_id']}",
        },
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    HfApi = require_hf_api()
    api = HfApi(token=token or None)
    try:
        identity = api.whoami(token=token or None)
        api.create_repo(
            repo_id=str(config["repo_id"]),
            repo_type=str(config["repo_type"]),
            private=bool(config["private"]),
            exist_ok=True,
        )
    except Exception as exc:  # pragma: no cover - network / API failure
        raise SystemExit(f"Failed to validate or create Hugging Face repo {config['repo_id']!r}: {exc}") from exc

    if isinstance(identity, dict):
        payload["hf_upload"]["authenticated_as"] = str(
            identity.get("name") or identity.get("fullname") or identity.get("email") or ""
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
