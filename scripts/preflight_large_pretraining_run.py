from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.corpus_manifest import iter_pretraining_document_rows
from training.packed_lm_dataset import read_packed_manifest
from training.token_packer import read_document_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight a document or packed pretraining manifest against the repo's "
            "current large-run design target: dominant Pgen, bounded Ptraj, and "
            "holdout-only Pbench."
        )
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--min-train-pgen-fraction", type=float, default=0.70)
    parser.add_argument("--min-train-ptraj-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-ptraj-fraction", type=float, default=0.30)
    parser.add_argument("--require-any-holdout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-holdout-groups", nargs="+", default=[])
    parser.add_argument("--json-output", type=str, default="")
    return parser.parse_args()


def _fraction(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _band_weight_summary_from_document_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    document_files = dict(manifest.get("document_files", {}))
    summary: dict[str, dict[str, int]] = {}
    for split, file_path in sorted(document_files.items()):
        band_bytes: Counter[str] = Counter()
        band_docs: Counter[str] = Counter()
        for row in iter_pretraining_document_rows(file_path):
            band = str(row.get("band", "pgen"))
            band_docs[band] += 1
            band_bytes[band] += int(row.get("byte_length", 0))
        summary[split] = {
            "total_bytes": int(sum(band_bytes.values())),
            "total_documents": int(sum(band_docs.values())),
            "band_bytes": dict(sorted(band_bytes.items())),
            "band_documents": dict(sorted(band_docs.items())),
        }
    return summary


def _document_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    manifest = read_document_manifest(manifest_path)
    split_summary = _band_weight_summary_from_document_manifest(manifest)
    holdout_counts = dict(manifest.get("holdout_counts", {}))
    holdout_groups = sorted(group for group in holdout_counts if group != "total" and holdout_counts[group] > 0)
    return {
        "kind": "document_manifest",
        "path": str(manifest_path),
        "train_weight_unit": "bytes",
        "split_summary": split_summary,
        "holdout_counts": holdout_counts,
        "holdout_groups": holdout_groups,
        "source_name": manifest.get("source_name", ""),
        "corpus_name": manifest.get("corpus_name", ""),
    }


def _packed_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    manifest = read_packed_manifest(manifest_path)
    splits = dict(manifest.get("splits", {}))
    split_summary: dict[str, dict[str, int]] = {}
    for split, payload in sorted(splits.items()):
        payload_dict = dict(payload)
        split_summary[split] = {
            "total_tokens": int(payload_dict.get("document_token_count", 0)),
            "total_documents": int(payload_dict.get("document_count", 0)),
            "band_tokens": {
                str(name): int(value)
                for name, value in sorted(dict(payload_dict.get("band_document_token_counts", {})).items())
            },
            "band_documents": {
                str(name): int(value)
                for name, value in sorted(dict(payload_dict.get("band_document_counts", {})).items())
            },
        }
    holdout_counts = dict(manifest.get("document_manifest_holdout_counts", {}))
    holdout_groups = sorted(group for group in holdout_counts if group != "total" and holdout_counts[group] > 0)
    return {
        "kind": "packed_manifest",
        "path": str(manifest_path),
        "train_weight_unit": "tokens",
        "split_summary": split_summary,
        "holdout_counts": holdout_counts,
        "holdout_groups": holdout_groups,
        "source_name": manifest.get("document_manifest_source_name", ""),
        "corpus_name": manifest.get("document_manifest_corpus_name", ""),
    }


def _payload_for_manifest(manifest_path: Path) -> dict[str, Any]:
    raw = _read_json(manifest_path)
    if "document_files" in raw:
        return _document_manifest_payload(manifest_path)
    if "splits" in raw and "document_manifest_path" in raw:
        return _packed_manifest_payload(manifest_path)
    raise SystemExit(f"{manifest_path} is neither a document manifest nor a packed manifest.")


def _band_weights(split_payload: dict[str, int], *, unit: str) -> dict[str, int]:
    if unit == "bytes":
        return {
            str(name): int(value)
            for name, value in sorted(dict(split_payload.get("band_bytes", {})).items())
        }
    if unit == "tokens":
        return {
            str(name): int(value)
            for name, value in sorted(dict(split_payload.get("band_tokens", {})).items())
        }
    raise ValueError(f"Unsupported weight unit: {unit}")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    payload = _payload_for_manifest(manifest_path)
    split_summary = dict(payload["split_summary"])
    train_summary = dict(split_summary.get("train", {}))
    val_summary = dict(split_summary.get("val", {}))
    weight_unit = str(payload["train_weight_unit"])

    train_band_weights = _band_weights(train_summary, unit=weight_unit)
    val_band_weights = _band_weights(val_summary, unit=weight_unit)
    train_total_weight = int(train_summary.get(f"total_{weight_unit}", 0))
    val_total_weight = int(val_summary.get(f"total_{weight_unit}", 0))

    train_pgen_fraction = _fraction(train_band_weights.get("pgen", 0), train_total_weight)
    train_ptraj_fraction = _fraction(train_band_weights.get("ptraj", 0), train_total_weight)
    train_pbench_fraction = _fraction(train_band_weights.get("pbench", 0), train_total_weight)
    val_pbench_fraction = _fraction(val_band_weights.get("pbench", 0), val_total_weight)

    failures: list[str] = []
    warnings: list[str] = []

    if train_total_weight <= 0:
        failures.append("train split has zero total weight")
    if train_band_weights.get("pgen", 0) <= 0:
        failures.append("train split is missing pgen entirely")
    if train_band_weights.get("ptraj", 0) <= 0:
        failures.append("train split is missing ptraj entirely")
    if train_pgen_fraction < float(args.min_train_pgen_fraction):
        failures.append(
            f"train pgen fraction {train_pgen_fraction:.4f} is below required minimum {float(args.min_train_pgen_fraction):.4f}"
        )
    if train_ptraj_fraction < float(args.min_train_ptraj_fraction):
        failures.append(
            f"train ptraj fraction {train_ptraj_fraction:.4f} is below required minimum {float(args.min_train_ptraj_fraction):.4f}"
        )
    if train_ptraj_fraction > float(args.max_train_ptraj_fraction):
        failures.append(
            f"train ptraj fraction {train_ptraj_fraction:.4f} exceeds allowed maximum {float(args.max_train_ptraj_fraction):.4f}"
        )
    if train_pbench_fraction > 0.0:
        failures.append(f"train split contains pbench weight {train_pbench_fraction:.4f}; pbench must remain holdout-only")
    if val_pbench_fraction > 0.0:
        failures.append(f"val split contains pbench weight {val_pbench_fraction:.4f}; pbench must remain holdout-only")

    holdout_counts = dict(payload.get("holdout_counts", {}))
    holdout_total = int(holdout_counts.get("total", 0))
    holdout_groups = list(payload.get("holdout_groups", []))
    if args.require_any_holdout and holdout_total <= 0:
        failures.append("manifest has no holdout benchmark rows; expected pbench holdouts for a large run")

    expected_holdout_groups = tuple(str(group) for group in args.expected_holdout_groups)
    missing_holdout_groups = [group for group in expected_holdout_groups if group not in holdout_groups]
    if missing_holdout_groups:
        failures.append(f"missing expected holdout groups: {', '.join(missing_holdout_groups)}")

    if train_pgen_fraction > 0.95:
        warnings.append("train pgen fraction exceeds 0.95; ptraj may be too weak to move reasoning behavior")
    if train_ptraj_fraction > 0.20:
        warnings.append("train ptraj fraction is above 0.20; monitor retention on native language tasks carefully")
    if holdout_total > 0 and len(holdout_groups) < 2:
        warnings.append("only one holdout benchmark group is present; broad benchmark tracking will be limited")

    result = {
        "manifest_path": str(manifest_path),
        "manifest_kind": payload["kind"],
        "source_name": payload.get("source_name", ""),
        "corpus_name": payload.get("corpus_name", ""),
        "train_weight_unit": weight_unit,
        "train_total_weight": train_total_weight,
        "val_total_weight": val_total_weight,
        "train_band_weights": train_band_weights,
        "val_band_weights": val_band_weights,
        "train_band_fractions": {
            "pgen": round(train_pgen_fraction, 6),
            "ptraj": round(train_ptraj_fraction, 6),
            "pbench": round(train_pbench_fraction, 6),
        },
        "val_band_fractions": {
            "pgen": round(_fraction(val_band_weights.get("pgen", 0), val_total_weight), 6),
            "ptraj": round(_fraction(val_band_weights.get("ptraj", 0), val_total_weight), 6),
            "pbench": round(val_pbench_fraction, 6),
        },
        "holdout_total": holdout_total,
        "holdout_groups": holdout_groups,
        "checks": {
            "min_train_pgen_fraction": float(args.min_train_pgen_fraction),
            "min_train_ptraj_fraction": float(args.min_train_ptraj_fraction),
            "max_train_ptraj_fraction": float(args.max_train_ptraj_fraction),
            "require_any_holdout": bool(args.require_any_holdout),
            "expected_holdout_groups": list(expected_holdout_groups),
        },
        "warnings": warnings,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        output_path = Path(args.json_output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
