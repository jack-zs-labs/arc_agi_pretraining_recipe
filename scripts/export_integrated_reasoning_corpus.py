from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_integrated_reasoning_stack import (
    build_data_only_payload,
    build_dataset_bundle,
    export_dataset_bundle,
    parse_args,
    write_data_only_csv,
    write_payload,
)


def main() -> None:
    args = parse_args()
    if not args.export_dir:
        raise SystemExit("Pass --export-dir to write the offline corpus manifest and token window files.")
    if args.corpus_manifest:
        raise SystemExit("The export entrypoint builds a fresh corpus bundle; do not pass --corpus-manifest.")
    if args.tokenizer in {"epiplex", "rust_bpe"} and not args.tokenizer_load and not args.tokenizer_save:
        args.tokenizer_save = str((Path(args.export_dir).resolve() / "reasoning_tokenizer.json"))

    bundle = build_dataset_bundle(args)
    manifest_path = export_dataset_bundle(
        args=args,
        bundle=bundle,
        export_dir=Path(args.export_dir).resolve(),
    )
    payload = build_data_only_payload(args, bundle)
    payload["export_manifest"] = str(manifest_path)

    write_payload(args, payload)
    write_data_only_csv(args, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
