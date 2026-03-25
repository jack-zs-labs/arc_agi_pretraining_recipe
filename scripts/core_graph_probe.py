from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.core_reasoning_adapter import build_core_reasoning_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the explicit CoRe graph extraction path.")
    parser.add_argument("--data-dir", type=str, default="arc_trajectory_sampler/data/core")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--dependency-kinds", nargs="+", choices=("control", "data", "infoflow"), default=None)
    parser.add_argument("--categories", nargs="+", choices=("trace", "list_source"), default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--graph-backend", choices=("auto", "heuristic", "python_ast"), default="auto")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--csv-output", type=str, default="")
    return parser.parse_args()


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    max_examples = None if args.max_examples <= 0 else args.max_examples
    tasks = build_core_reasoning_tasks(
        data_dir=args.data_dir,
        max_examples=max_examples,
        dependency_kinds=args.dependency_kinds,
        categories=args.categories,
        languages=args.languages,
        graph_backend=args.graph_backend,
    )

    rows: list[dict[str, object]] = []
    infoflow_edge_types: dict[str, int] = {}
    query_source_hits = 0
    query_target_hits = 0
    for task in tasks:
        graph_node_ids = {node.node_id for node in task.graph_nodes}
        query_source_hit = task.query_source is not None and (
            task.query_source.name is None and f"line:{task.query_source.line}" in graph_node_ids
            or task.query_source.name is not None and f"var:{task.query_source.name}@{task.query_source.line}" in graph_node_ids
        )
        query_target_hit = (
            task.query_target.name is None and f"line:{task.query_target.line}" in graph_node_ids
            or task.query_target.name is not None and f"var:{task.query_target.name}@{task.query_target.line}" in graph_node_ids
        )
        query_source_hits += int(bool(query_source_hit))
        query_target_hits += int(bool(query_target_hit))
        for edge_type in task.trace_edge_types:
            infoflow_edge_types[edge_type] = infoflow_edge_types.get(edge_type, 0) + 1
        rows.append(
            {
                "task_id": task.task_id,
                "dependency_kind": task.dependency_kind,
                "category": task.category,
                "graph_backend": task.graph_backend,
                "graph_nodes": len(task.graph_nodes),
                "graph_edges": len(task.graph_edges),
                "positive_sources": len(task.positive_sources),
                "trace_nodes": len(task.trace_nodes),
                "trace_edge_types": "|".join(task.trace_edge_types),
                "query_source_graph_hit": bool(query_source_hit),
                "query_target_graph_hit": bool(query_target_hit),
            }
        )

    summary = {
        "task_count": len(tasks),
        "dependency_kind_counts": {
            dependency_kind: sum(1 for task in tasks if task.dependency_kind == dependency_kind)
            for dependency_kind in ("control", "data", "infoflow")
        },
        "category_counts": {
            category: sum(1 for task in tasks if task.category == category)
            for category in ("trace", "list_source")
        },
        "graph_backend_counts": {
            backend: sum(1 for task in tasks if task.graph_backend == backend)
            for backend in sorted({task.graph_backend for task in tasks})
        },
        "mean_graph_nodes": statistics.mean(row["graph_nodes"] for row in rows) if rows else 0.0,
        "mean_graph_edges": statistics.mean(row["graph_edges"] for row in rows) if rows else 0.0,
        "mean_positive_sources": statistics.mean(row["positive_sources"] for row in rows) if rows else 0.0,
        "mean_trace_nodes": statistics.mean(row["trace_nodes"] for row in rows) if rows else 0.0,
        "query_source_graph_hit_rate": (query_source_hits / len(tasks)) if tasks else 0.0,
        "query_target_graph_hit_rate": (query_target_hits / len(tasks)) if tasks else 0.0,
        "infoflow_trace_edge_types": infoflow_edge_types,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.csv_output, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
