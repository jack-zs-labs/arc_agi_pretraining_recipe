from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import re
from typing import Sequence

from .core_graph_extractor import (
    CoreCodeGraph,
    CoreGraphEdge,
    CoreGraphNode,
    GRAPH_BACKEND_CHOICES,
    cfg_edges_for_graph,
    dfg_edges_for_graph,
    extract_core_code_graph,
    graph_edges_json,
    graph_nodes_json,
    ifg_edges_for_graph,
    line_node_id,
    var_def_node_id,
)
from .core_loader import (
    CORERow,
    CoreCodeLine,
    CoreNodeRef,
    SOURCE_OUTPUT_KEYS,
    TRACE_BOOLEAN_OUTPUT_KEYS,
    extract_core_code_lines,
    extract_core_question,
    load_core_rows,
)
from .state_adapter import FORMAT_HEADER


_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9_.:@/-]+")


@dataclass(frozen=True)
class CoreReasoningTask:
    task_id: str
    dependency_kind: str
    category: str
    dataset: str
    language: str
    function_name: str
    start_line: int
    end_line: int
    question: str
    query_source: CoreNodeRef | None
    query_target: CoreNodeRef
    candidate_sources: tuple[CoreNodeRef, ...]
    positive_sources: tuple[CoreNodeRef, ...]
    trace_nodes: tuple[CoreNodeRef, ...]
    trace_edge_types: tuple[str, ...]
    direct_edges: tuple[tuple[CoreNodeRef, CoreNodeRef], ...]
    code_lines: tuple[CoreCodeLine, ...]
    graph_nodes: tuple[CoreGraphNode, ...]
    graph_edges: tuple[CoreGraphEdge, ...]
    cfg_edges: tuple[CoreGraphEdge, ...]
    dfg_edges: tuple[CoreGraphEdge, ...]
    ifg_edges: tuple[CoreGraphEdge, ...]
    graph_backend: str
    target_answer: dict[str, object]
    auxiliary_targets: dict[str, object]
    metadata: dict[str, object]

    @property
    def family(self) -> str:
        return f"core_{self.dependency_kind}_{self.category}"

    @property
    def trace_step(self) -> str:
        return f"{self.dependency_kind}_{self.category}"


def snippet_identity(row: CORERow) -> tuple[str, str, str, str, int, int]:
    return (row.dataset, row.pid, row.sid, row.funname, row.start, row.end)


def trace_graph_key(row: CORERow) -> tuple[str, str, str, str, str, int, int]:
    return (row.dependency_kind,) + snippet_identity(row)


def source_query_key(row: CORERow) -> tuple[str, str, str, str, str, int, int, CoreNodeRef]:
    return trace_graph_key(row) + (row.dst,)


def node_sort_key(node: CoreNodeRef) -> tuple[int, str, str]:
    return (node.line, node.name or "", node.use_kind or "")


def compact_token(text: str) -> str:
    return _NON_ALNUM_RE.sub("_", text.strip()).strip("_") or "empty"


def node_token(node: CoreNodeRef) -> str:
    if node.name is None:
        return f"line:{node.line}"
    if node.use_kind is None:
        return f"{node.name}@{node.line}"
    return f"{node.name}@{node.line}:{node.use_kind}"


def node_json(node: CoreNodeRef | None) -> dict[str, object] | None:
    if node is None:
        return None
    return node.to_jsonable()


def node_benchmark_value(node: CoreNodeRef) -> int | list[object]:
    return node.to_benchmark_value()


def line_only(node: CoreNodeRef) -> CoreNodeRef:
    return CoreNodeRef(line=node.line)


def graph_node_id_for_ref(node: CoreNodeRef) -> str:
    if node.name is None:
        return line_node_id(node.line)
    return var_def_node_id(node.name, node.line)


def transitive_reduction_edges(
    pairs: set[tuple[CoreNodeRef, CoreNodeRef]],
) -> tuple[tuple[CoreNodeRef, CoreNodeRef], ...]:
    by_source: dict[CoreNodeRef, set[CoreNodeRef]] = defaultdict(set)
    for source, target in pairs:
        by_source[source].add(target)
    reduced: list[tuple[CoreNodeRef, CoreNodeRef]] = []
    for source, target in sorted(pairs, key=lambda pair: (node_sort_key(pair[0]), node_sort_key(pair[1]))):
        intermediates = by_source.get(source, set())
        has_bridge = any(
            middle != target and (middle, target) in pairs
            for middle in intermediates
        )
        if not has_bridge:
            reduced.append((source, target))
    return tuple(reduced)


def shortest_path(
    edges: Sequence[tuple[CoreNodeRef, CoreNodeRef]],
    source: CoreNodeRef,
    target: CoreNodeRef,
) -> tuple[CoreNodeRef, ...]:
    adjacency: dict[CoreNodeRef, list[CoreNodeRef]] = defaultdict(list)
    for left, right in edges:
        adjacency[left].append(right)
    queue: deque[tuple[CoreNodeRef, tuple[CoreNodeRef, ...]]] = deque([(source, (source,))])
    seen = {source}
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for neighbor in adjacency.get(node, ()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, path + (neighbor,)))
    return ()


def build_trace_edges_payload(
    dependency_kind: str,
    nodes: Sequence[CoreNodeRef],
    edge_types: Sequence[str],
) -> list[dict[str, object]] | list[int]:
    if dependency_kind == "control":
        return [node.line for node in nodes]
    if dependency_kind == "data":
        return [
            {
                "from": node_benchmark_value(nodes[index]),
                "to": node_benchmark_value(nodes[index + 1]),
            }
            for index in range(len(nodes) - 1)
        ]
    return [
        {
            "from": node_benchmark_value(nodes[index]),
            "to": node_benchmark_value(nodes[index + 1]),
            "type": edge_types[index],
        }
        for index in range(len(nodes) - 1)
    ]


def build_trace_target_answer(
    *,
    dependency_kind: str,
    query_positive: bool,
    trace_nodes: Sequence[CoreNodeRef],
    trace_edge_types: Sequence[str],
) -> dict[str, object]:
    payload: dict[str, object] = {TRACE_BOOLEAN_OUTPUT_KEYS[dependency_kind]: query_positive}
    if query_positive:
        payload["Trace"] = build_trace_edges_payload(
            dependency_kind,
            trace_nodes,
            trace_edge_types,
        )
    return payload


def build_source_target_answer(
    *,
    dependency_kind: str,
    positive_sources: Sequence[CoreNodeRef],
) -> dict[str, object]:
    source_key = SOURCE_OUTPUT_KEYS[dependency_kind]
    if dependency_kind == "infoflow" and not positive_sources:
        return {source_key: False}
    return {source_key: [node_benchmark_value(node) for node in positive_sources]}


def infoflow_edge_type(
    source: CoreNodeRef,
    target: CoreNodeRef,
    *,
    data_direct_edges: set[tuple[CoreNodeRef, CoreNodeRef]],
    control_direct_edges: set[tuple[CoreNodeRef, CoreNodeRef]],
    graph: CoreCodeGraph,
) -> str:
    if (source, target) in data_direct_edges:
        return "data"
    source_node_id = graph_node_id_for_ref(source)
    target_node_id = graph_node_id_for_ref(target)
    target_line_id = line_node_id(target.line)
    candidate_data_pairs = {
        (edge.source_id, edge.target_id)
        for edge in graph.edges
        if edge.edge_type == "candidate_data"
    }
    condition_use_pairs = {
        (edge.source_id, edge.target_id)
        for edge in graph.edges
        if edge.edge_type == "condition_use"
    }
    controls_line_pairs = {
        (edge.source_id, edge.target_id)
        for edge in graph.edges
        if edge.edge_type == "controls_line"
    }
    if (source_node_id, target_node_id) in candidate_data_pairs:
        return "data"
    if any(
        condition_source == source_node_id and (condition_line_id, target_line_id) in controls_line_pairs
        for condition_source, condition_line_id in condition_use_pairs
    ):
        return "control"
    if (line_only(source), line_only(target)) in control_direct_edges:
        return "control"
    return "control"


def auxiliary_targets_for_task(
    *,
    dependency_kind: str,
    category: str,
    graph_backend: str,
    query_source: CoreNodeRef | None,
    query_target: CoreNodeRef,
    query_positive: bool,
    candidate_sources: Sequence[CoreNodeRef],
    positive_sources: Sequence[CoreNodeRef],
    trace_nodes: Sequence[CoreNodeRef],
    trace_edge_types: Sequence[str],
    direct_edges: Sequence[tuple[CoreNodeRef, CoreNodeRef]],
    cfg_edges: Sequence[CoreGraphEdge],
    dfg_edges: Sequence[CoreGraphEdge],
    ifg_edges: Sequence[CoreGraphEdge],
) -> dict[str, object]:
    return {
        "dependency_kind": dependency_kind,
        "category": category,
        "graph_backend": graph_backend,
        "query_positive": query_positive,
        "query_source": node_json(query_source),
        "query_target": node_json(query_target),
        "candidate_sources": [node.to_jsonable() for node in candidate_sources],
        "source_set": [node.to_jsonable() for node in positive_sources],
        "direct_dependency_edges": [
            {
                "from": left.to_jsonable(),
                "to": right.to_jsonable(),
            }
            for left, right in direct_edges
        ],
        "trace_nodes": [node.to_jsonable() for node in trace_nodes],
        "trace_edge_types": list(trace_edge_types),
        "cfg_edge_count": len(cfg_edges),
        "dfg_edge_count": len(dfg_edges),
        "ifg_edge_count": len(ifg_edges),
        "infoflow_has_data_edge": any(edge_type == "data" for edge_type in trace_edge_types),
    }


def difficulty_for_task(task: CoreReasoningTask) -> int:
    score = 1
    score += min(2, len(task.code_lines) // 25)
    score += min(1, len(task.positive_sources) // 4)
    score += min(1, len(task.graph_edges) // 40)
    score += min(1, len(task.ifg_edges) // 60)
    if task.dependency_kind == "infoflow":
        score += 1
    if task.category == "trace" and len(task.trace_nodes) >= 3:
        score += 1
    return min(score, 5)


def state_tokens_for_task(task: CoreReasoningTask) -> tuple[str, ...]:
    tokens = [
        f"dep:{task.dependency_kind}",
        f"category:{task.category}",
        f"lang:{compact_token(task.language)}",
        f"fn:{compact_token(task.function_name)}",
        f"target:{node_token(task.query_target)}",
        f"source_count:{len(task.positive_sources)}",
        f"candidate_count:{len(task.candidate_sources)}",
        f"line_count:{len(task.code_lines)}",
        f"graph_nodes:{len(task.graph_nodes)}",
        f"graph_edges:{len(task.graph_edges)}",
        f"cfg_edges:{len(task.cfg_edges)}",
        f"dfg_edges:{len(task.dfg_edges)}",
        f"ifg_edges:{len(task.ifg_edges)}",
        f"graph_backend:{compact_token(task.graph_backend)}",
    ]
    if task.query_source is not None:
        tokens.append(f"source:{node_token(task.query_source)}")
    tokens.extend(f"support:{node_token(node)}" for node in task.positive_sources[:12])
    tokens.extend(f"trace:{node_token(node)}" for node in task.trace_nodes[:8])
    for code_line in task.code_lines[:24]:
        tokens.append(f"line:{code_line.line}")
        tokens.append(f"code:{compact_token(code_line.text)}")
    return tuple(tokens)


def state_scalars_for_task(task: CoreReasoningTask) -> tuple[float, ...]:
    source_line = 0.0 if task.query_source is None else float(task.query_source.line)
    return (
        float(len(task.code_lines)),
        float(len(task.candidate_sources)),
        float(len(task.positive_sources)),
        float(len(task.trace_nodes)),
        float(len(task.graph_nodes)),
        float(len(task.graph_edges)),
        float(len(task.cfg_edges)),
        float(len(task.dfg_edges)),
        float(len(task.ifg_edges)),
        source_line,
        float(task.query_target.line),
    )


def serialize_core_reasoning_task(task: CoreReasoningTask) -> str:
    state_tokens = state_tokens_for_task(task)
    state_scalars = state_scalars_for_task(task)
    serialized_graph_nodes = graph_nodes_json(task.graph_nodes)
    serialized_graph_edges = graph_edges_json(task.graph_edges)
    serialized_cfg_edges = graph_edges_json(task.cfg_edges)
    serialized_dfg_edges = graph_edges_json(task.dfg_edges)
    serialized_ifg_edges = graph_edges_json(task.ifg_edges)
    serialized_graph_edges.extend(
        {
            "type": f"benchmark_{task.dependency_kind}_direct",
            "from": left.to_jsonable(),
            "to": right.to_jsonable(),
            "attributes": {},
        }
        for left, right in task.direct_edges
    )
    lines = [
        FORMAT_HEADER,
        "record_type=terminal_answer",
        "dataset=core",
        "source_modality=code_static_analysis",
        f"family={task.family}",
        f"difficulty={difficulty_for_task(task)}",
        f"trajectory_id={task.task_id}",
        "step_index=0",
        "trace_length=1",
        f"trace_step={task.trace_step}",
        "previous_action=<start>",
        f"candidate_bucket=family:{task.family}|step:0|prev:<start>",
        "answer_format=json",
        "variant_kind=benchmark_query",
        "trajectory_role=gold",
        f"dependency_kind={task.dependency_kind}",
        f"task_category={task.category}",
        f"language={task.language}",
        f"source_dataset={task.dataset}",
        f"function={task.function_name}",
        f"code_span={task.start_line}:{task.end_line}",
        f"graph_backend={task.graph_backend}",
        "state_tokens=" + " ".join(state_tokens),
        "state_scalars=" + ",".join(f"{value:.6f}" for value in state_scalars),
        "question=" + json.dumps(task.question, separators=(",", ":"), sort_keys=True),
        "query_source=" + json.dumps(node_json(task.query_source), separators=(",", ":"), sort_keys=True),
        "query_target=" + json.dumps(node_json(task.query_target), separators=(",", ":"), sort_keys=True),
        "candidate_sources="
        + json.dumps([node.to_jsonable() for node in task.candidate_sources], separators=(",", ":"), sort_keys=True),
        "positive_sources="
        + json.dumps([node.to_jsonable() for node in task.positive_sources], separators=(",", ":"), sort_keys=True),
        "code_lines="
        + json.dumps(
            [{"line": line.line, "text": line.text} for line in task.code_lines],
            separators=(",", ":"),
            sort_keys=True,
        ),
        "graph_nodes=" + json.dumps(serialized_graph_nodes, separators=(",", ":"), sort_keys=True),
        "graph_edges=" + json.dumps(serialized_graph_edges, separators=(",", ":"), sort_keys=True),
        "cfg_edges=" + json.dumps(serialized_cfg_edges, separators=(",", ":"), sort_keys=True),
        "dfg_edges=" + json.dumps(serialized_dfg_edges, separators=(",", ":"), sort_keys=True),
        "ifg_edges=" + json.dumps(serialized_ifg_edges, separators=(",", ":"), sort_keys=True),
    ]
    if task.trace_nodes:
        lines.append(
            "trace_nodes="
            + json.dumps([node.to_jsonable() for node in task.trace_nodes], separators=(",", ":"), sort_keys=True)
        )
    if task.trace_edge_types:
        lines.append("trace_edge_types=" + json.dumps(list(task.trace_edge_types), separators=(",", ":"), sort_keys=True))
    lines.append("auxiliary_targets=" + json.dumps(task.auxiliary_targets, separators=(",", ":"), sort_keys=True))
    lines.append("target_answer=" + json.dumps(task.target_answer, separators=(",", ":"), sort_keys=True))
    return "\n".join(lines) + "\n"


def build_core_reasoning_tasks(
    *,
    data_dir: str,
    max_examples: int | None = None,
    languages: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    dependency_kinds: Sequence[str] | None = None,
    graph_backend: str = "auto",
) -> tuple[CoreReasoningTask, ...]:
    if graph_backend not in GRAPH_BACKEND_CHOICES:
        raise ValueError(f"Unsupported CoRe graph backend: {graph_backend!r}")
    rows = load_core_rows(
        data_dir=data_dir,
        max_rows=None,
        languages=languages,
        categories=categories,
        dependency_kinds=dependency_kinds,
    )
    return compile_core_rows(rows, max_examples=max_examples, graph_backend=graph_backend)


def compile_core_rows(
    rows: Sequence[CORERow],
    *,
    max_examples: int | None = None,
    graph_backend: str = "auto",
) -> tuple[CoreReasoningTask, ...]:
    positive_trace_pairs: dict[tuple[str, str, str, str, str, int, int], set[tuple[CoreNodeRef, CoreNodeRef]]] = defaultdict(set)
    positive_trace_sources: dict[tuple[str, str, str, str, str, int, int, CoreNodeRef], set[CoreNodeRef]] = defaultdict(set)
    source_rows: dict[tuple[str, str, str, str, str, int, int, CoreNodeRef], list[CORERow]] = defaultdict(list)
    trace_rows: list[CORERow] = []

    for row in rows:
        if row.category == "trace":
            trace_rows.append(row)
            if row.groundtruth:
                positive_trace_pairs[trace_graph_key(row)].add((row.src, row.dst))
                positive_trace_sources[source_query_key(row)].add(row.src)
        elif row.category == "list_source":
            source_rows[source_query_key(row)].append(row)

    direct_edges_by_graph = {
        key: transitive_reduction_edges(pairs)
        for key, pairs in positive_trace_pairs.items()
    }
    data_direct_edge_sets = {
        key: set(edges)
        for key, edges in direct_edges_by_graph.items()
        if key[0] == "data"
    }
    control_direct_edge_sets = {
        key: {(line_only(left), line_only(right)) for left, right in edges}
        for key, edges in direct_edges_by_graph.items()
        if key[0] == "control"
    }
    graph_cache: dict[tuple[str, str, str, str, int, int], CoreCodeGraph] = {}
    exemplar_rows: dict[tuple[str, str, str, str, int, int], CORERow] = {}
    for row in rows:
        identity = snippet_identity(row)
        exemplar_rows.setdefault(identity, row)
    for identity, exemplar in exemplar_rows.items():
        graph_cache[identity] = extract_core_code_graph(
            extract_core_code_lines(exemplar.prompt),
            language=exemplar.language,
            backend=graph_backend,
        )

    tasks: list[CoreReasoningTask] = []

    for row in sorted(trace_rows, key=lambda item: item.task_id):
        graph_key = trace_graph_key(row)
        graph = graph_cache[snippet_identity(row)]
        cfg_edges = cfg_edges_for_graph(graph)
        dfg_edges = dfg_edges_for_graph(graph)
        ifg_edges = ifg_edges_for_graph(graph)
        list_key = source_query_key(row)
        list_query_rows = source_rows.get(list_key, [])
        candidate_sources = tuple(
            sorted(
                {candidate.src for candidate in list_query_rows} or positive_trace_sources.get(list_key, {row.src}),
                key=node_sort_key,
            )
        )
        positive_sources = tuple(
            sorted(
                {candidate.src for candidate in list_query_rows if candidate.groundtruth}
                or positive_trace_sources.get(list_key, set()),
                key=node_sort_key,
            )
        )
        direct_edges = direct_edges_by_graph.get(graph_key, ())
        trace_nodes = ()
        trace_edge_types: tuple[str, ...] = ()
        if row.groundtruth:
            trace_nodes = shortest_path(direct_edges, row.src, row.dst) or (row.src, row.dst)
            if row.dependency_kind == "infoflow":
                data_key = ("data",) + graph_key[1:]
                control_key = ("control",) + graph_key[1:]
                edge_types = [
                    infoflow_edge_type(
                        trace_nodes[index],
                        trace_nodes[index + 1],
                        data_direct_edges=data_direct_edge_sets.get(data_key, set()),
                        control_direct_edges=control_direct_edge_sets.get(control_key, set()),
                        graph=graph,
                    )
                    for index in range(len(trace_nodes) - 1)
                ]
                trace_edge_types = tuple(edge_types)
        target_answer = build_trace_target_answer(
            dependency_kind=row.dependency_kind,
            query_positive=row.groundtruth,
            trace_nodes=trace_nodes,
            trace_edge_types=trace_edge_types,
        )
        tasks.append(
            CoreReasoningTask(
                task_id=row.task_id,
                dependency_kind=row.dependency_kind,
                category=row.category,
                dataset=row.dataset,
                language=row.language,
                function_name=row.funname,
                start_line=row.start,
                end_line=row.end,
                question=extract_core_question(row.prompt),
                query_source=row.src,
                query_target=row.dst,
                candidate_sources=candidate_sources,
                positive_sources=positive_sources,
                trace_nodes=trace_nodes,
                trace_edge_types=trace_edge_types,
                direct_edges=direct_edges,
                code_lines=extract_core_code_lines(row.prompt),
                graph_nodes=graph.nodes,
                graph_edges=graph.edges,
                cfg_edges=cfg_edges,
                dfg_edges=dfg_edges,
                ifg_edges=ifg_edges,
                graph_backend=graph.backend,
                target_answer=target_answer,
                auxiliary_targets=auxiliary_targets_for_task(
                    dependency_kind=row.dependency_kind,
                    category=row.category,
                    graph_backend=graph.backend,
                    query_source=row.src,
                    query_target=row.dst,
                    query_positive=row.groundtruth,
                    candidate_sources=candidate_sources,
                    positive_sources=positive_sources,
                    trace_nodes=trace_nodes,
                    trace_edge_types=trace_edge_types,
                    direct_edges=direct_edges,
                    cfg_edges=cfg_edges,
                    dfg_edges=dfg_edges,
                    ifg_edges=ifg_edges,
                ),
                metadata={
                    "source_count": len(positive_sources),
                    "candidate_count": len(candidate_sources),
                    "query_positive": row.groundtruth,
                    "graph_backend": graph.backend,
                },
            )
        )

    for list_key, grouped_rows in sorted(source_rows.items(), key=lambda item: item[0]):
        exemplar = grouped_rows[0]
        graph_key = trace_graph_key(exemplar)
        graph = graph_cache[snippet_identity(exemplar)]
        cfg_edges = cfg_edges_for_graph(graph)
        dfg_edges = dfg_edges_for_graph(graph)
        ifg_edges = ifg_edges_for_graph(graph)
        positive_sources = tuple(
            sorted({row.src for row in grouped_rows if row.groundtruth} or positive_trace_sources.get(list_key, set()), key=node_sort_key)
        )
        candidate_sources = tuple(sorted({row.src for row in grouped_rows}, key=node_sort_key))
        target_answer = build_source_target_answer(
            dependency_kind=exemplar.dependency_kind,
            positive_sources=positive_sources,
        )
        tasks.append(
            CoreReasoningTask(
                task_id=f"{exemplar.task_id}_aggregate",
                dependency_kind=exemplar.dependency_kind,
                category=exemplar.category,
                dataset=exemplar.dataset,
                language=exemplar.language,
                function_name=exemplar.funname,
                start_line=exemplar.start,
                end_line=exemplar.end,
                question=extract_core_question(exemplar.prompt),
                query_source=None,
                query_target=exemplar.dst,
                candidate_sources=candidate_sources,
                positive_sources=positive_sources,
                trace_nodes=(),
                trace_edge_types=(),
                direct_edges=direct_edges_by_graph.get(graph_key, ()),
                code_lines=extract_core_code_lines(exemplar.prompt),
                graph_nodes=graph.nodes,
                graph_edges=graph.edges,
                cfg_edges=cfg_edges,
                dfg_edges=dfg_edges,
                ifg_edges=ifg_edges,
                graph_backend=graph.backend,
                target_answer=target_answer,
                auxiliary_targets=auxiliary_targets_for_task(
                    dependency_kind=exemplar.dependency_kind,
                    category=exemplar.category,
                    graph_backend=graph.backend,
                    query_source=None,
                    query_target=exemplar.dst,
                    query_positive=bool(positive_sources),
                    candidate_sources=candidate_sources,
                    positive_sources=positive_sources,
                    trace_nodes=(),
                    trace_edge_types=(),
                    direct_edges=direct_edges_by_graph.get(graph_key, ()),
                    cfg_edges=cfg_edges,
                    dfg_edges=dfg_edges,
                    ifg_edges=ifg_edges,
                ),
                metadata={
                    "source_count": len(positive_sources),
                    "candidate_count": len(candidate_sources),
                    "query_positive": bool(positive_sources),
                    "graph_backend": graph.backend,
                },
            )
        )

    tasks.sort(key=lambda task: task.task_id)
    if max_examples is not None:
        tasks = tasks[:max_examples]
    return tuple(tasks)
