from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from .oscar_scope_corpus import OscarScopeRecord, build_oscar_scope_records, discover_oscar_scope_roots
from .state_adapter import FORMAT_HEADER


OSCAR_GRAPH_REASONING_BENCHMARK = "oscar_graph_reasoning"
OSCAR_GRAPH_REASONING_FAMILIES = (
    "oscar_graph_relation",
    "oscar_graph_neighbors",
    "oscar_graph_path_completion",
    "oscar_graph_grounding",
    "oscar_graph_executor_rollout",
    "oscar_graph_executor_trace",
)


@dataclass(frozen=True)
class OscarGraphNode:
    node_id: str
    label: str
    category: str
    domain: str
    description: str
    provenance: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class OscarGraphEdge:
    source_id: str
    relation: str
    target_id: str
    domain: str
    provenance: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class OscarGraphReasoningTask:
    task_id: str
    benchmark: str
    family: str
    trace_step: str
    text: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class OscarGraphExecutorMotif:
    motif_id: str
    motif_label: str
    graph_domain: str
    abstraction_level: str
    source_node_id: str
    rollout_node_ids: tuple[str, ...]
    rollout_relations: tuple[str, ...]
    cue_keywords: tuple[str, ...]


def _default_oscar_graph_sources() -> dict[str, tuple[dict[str, str], ...]]:
    roots = discover_oscar_scope_roots()
    root = roots[0] if roots else None

    def source_entry(relative_path: str, section: str) -> dict[str, str]:
        if root is None:
            return {"path": relative_path, "section": section}
        return {"path": str((root / relative_path).resolve()), "section": section}

    return {
        "primitive": (
            source_entry("spec/unified_process_intelligence_bugfix.tex", "The single primitive: a typed temporal property graph"),
            source_entry("spec/unified_process_intelligence.tex", "The single primitive: a typed temporal property graph"),
        ),
        "agents": (
            source_entry("spec/unified_process_intelligence_bugfix.tex", "Multi-tenant architecture: Oscar, Kurt, and the global model"),
            source_entry("spec/unified_process_intelligence.tex", "Multi-tenant architecture: Oscar, Kurt, and the global model"),
        ),
        "recursive": (
            source_entry("spec/recursive_hierarchical_addendum.tex", "Abstract"),
        ),
        "claim_graph": (
            source_entry("meeting_notes_for_spec/pe_workflow_design_integrated_coherent.tex", "Approval claim graph"),
        ),
    }


_GROUNDING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "process_graph_prefix": ("typed temporal property graph", "single primitive", "process intelligence graph prefix"),
    "object_type_family": ("object types", "persistent objects", "object-centric"),
    "event_type_family": ("event types", "timestamped events", "activities"),
    "context_type_family": ("context types", "evolving context", "typed temporal contexts"),
    "involves_edge": ("involves", "involves links"),
    "object_lifecycle": ("object lifecycle", "ordered lifecycle", "event set"),
    "activity_trace": ("activity trace", "label trace", "trace"),
    "milestone_time": ("milestone time", "milestone times"),
    "duration_metric": ("duration", "cycle time"),
    "oscar_agent": ("local agent oscar", "oscar state", "policy control"),
    "kurt_agent": ("local agent kurt", "kurt state", "observability"),
    "global_model": ("global foundation model", "practice compiler", "f_theta", "f_\\theta"),
    "intervention_templates": ("intervention templates", "intervention template"),
    "monitor_templates": ("monitoring templates", "monitor bundles", "monitor template"),
    "typed_temporal_context": ("typed temporal context", "temporal contexts"),
    "higher_layer_object": ("higher-layer object", "higher layer object"),
    "higher_layer_event": ("higher-layer event", "higher layer event"),
    "hierarchical_graph": ("graph-of-graphs", "hierarchical graph", "multi-scale reasoning"),
    "approval_claim_graph": ("approval claim graph", "acyclic claim graph", "approval-relevant"),
    "leaf_evidence_node": ("leaf evidence nodes", "leaf node", "emit a local e-process"),
    "same_null_merge_node": ("same-null merge nodes", "same local null", "same-null evidence"),
    "gate_node": ("gate nodes", "distinct blockers", "union-null"),
    "arbitrary_dependence": ("arbitrary dependence",),
    "independent_dependence": ("independent",),
    "sequential_dependence": ("sequential",),
    "weighted_mean_merge": ("weighted mean merge", "weighted average", "arbitrary dependence"),
    "product_merge": ("sequential product", "product process", "independent under the null"),
    "minimum_gate": ("minimum gate", "pointwise minimum", "gate-by-minimum"),
}


def build_oscar_canonical_graph() -> tuple[tuple[OscarGraphNode, ...], tuple[OscarGraphEdge, ...]]:
    sources = _default_oscar_graph_sources()

    def node(
        node_id: str,
        label: str,
        *,
        category: str,
        domain: str,
        description: str,
        source_key: str,
    ) -> OscarGraphNode:
        return OscarGraphNode(
            node_id=node_id,
            label=label,
            category=category,
            domain=domain,
            description=description,
            provenance=sources[source_key],
        )

    def edge(source_id: str, relation: str, target_id: str, *, domain: str, source_key: str) -> OscarGraphEdge:
        return OscarGraphEdge(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
            domain=domain,
            provenance=sources[source_key],
        )

    nodes = (
        node(
            "process_graph_prefix",
            "Process Intelligence Graph prefix",
            category="graph_primitive",
            domain="primitive",
            description="The single system-agnostic primitive: a typed temporal property graph prefix over nodes, edges, types, attributes, and event time.",
            source_key="primitive",
        ),
        node(
            "object_type_family",
            "Object node types",
            category="node_family",
            domain="primitive",
            description="The object partition Theta of persistent business or system objects.",
            source_key="primitive",
        ),
        node(
            "event_type_family",
            "Event node types",
            category="node_family",
            domain="primitive",
            description="The event partition Sigma of timestamped activities.",
            source_key="primitive",
        ),
        node(
            "context_type_family",
            "Context node types",
            category="node_family",
            domain="primitive",
            description="The context partition Xi of evolving contextual state.",
            source_key="primitive",
        ),
        node(
            "involves_edge",
            "INVOLVES edge",
            category="edge_type",
            domain="primitive",
            description="A canonical edge from an event node to an object node.",
            source_key="primitive",
        ),
        node(
            "object_lifecycle",
            "Object lifecycle",
            category="graph_functional",
            domain="primitive",
            description="The ordered event sequence involving one object.",
            source_key="primitive",
        ),
        node(
            "activity_trace",
            "Activity trace",
            category="graph_functional",
            domain="primitive",
            description="The ordered sequence of activity labels induced by an object lifecycle.",
            source_key="primitive",
        ),
        node(
            "milestone_time",
            "Milestone time",
            category="graph_functional",
            domain="primitive",
            description="The minimum event time over a chosen set of labels on an object lifecycle.",
            source_key="primitive",
        ),
        node(
            "duration_metric",
            "Duration metric",
            category="graph_functional",
            domain="primitive",
            description="A duration defined as the difference between two milestone times.",
            source_key="primitive",
        ),
        node(
            "oscar_agent",
            "Oscar agent",
            category="agent",
            domain="agents",
            description="The local process-intelligence and policy-control agent.",
            source_key="agents",
        ),
        node(
            "kurt_agent",
            "Kurt agent",
            category="agent",
            domain="agents",
            description="The local model-training, evaluation, and observability agent.",
            source_key="agents",
        ),
        node(
            "global_model",
            "Global foundation model F_theta",
            category="agent",
            domain="agents",
            description="The permission-aware global practice compiler and recommender.",
            source_key="agents",
        ),
        node(
            "intervention_templates",
            "Intervention template library",
            category="template_library",
            domain="agents",
            description="Portable intervention templates suggested to local agents.",
            source_key="agents",
        ),
        node(
            "monitor_templates",
            "Monitoring template library",
            category="template_library",
            domain="agents",
            description="Portable monitor bundles suggested to local agents.",
            source_key="agents",
        ),
        node(
            "typed_temporal_context",
            "Typed temporal context",
            category="recursive_unit",
            domain="recursive",
            description="A contextual slice that can be reified at a higher abstraction layer.",
            source_key="recursive",
        ),
        node(
            "higher_layer_object",
            "Higher-layer object",
            category="recursive_unit",
            domain="recursive",
            description="A higher-layer object created by reifying lower-layer context.",
            source_key="recursive",
        ),
        node(
            "higher_layer_event",
            "Higher-layer event",
            category="recursive_unit",
            domain="recursive",
            description="A higher-layer event created by reifying lower-layer context.",
            source_key="recursive",
        ),
        node(
            "hierarchical_graph",
            "Hierarchical graph-of-graphs",
            category="recursive_unit",
            domain="recursive",
            description="A multi-scale graph view preserving provenance across abstraction layers.",
            source_key="recursive",
        ),
        node(
            "approval_claim_graph",
            "Approval claim graph",
            category="claim_graph",
            domain="claim_graph",
            description="Acyclic graph of approval-relevant claims, merges, and gates extracted from the operational graph.",
            source_key="claim_graph",
        ),
        node(
            "leaf_evidence_node",
            "Leaf evidence node",
            category="claim_node_class",
            domain="claim_graph",
            description="A claim-graph node that ingests evidence and emits a local e-process.",
            source_key="claim_graph",
        ),
        node(
            "same_null_merge_node",
            "Same-null merge node",
            category="claim_node_class",
            domain="claim_graph",
            description="A claim-graph node whose children support the same local null and must be merged validly.",
            source_key="claim_graph",
        ),
        node(
            "gate_node",
            "Gate node",
            category="claim_node_class",
            domain="claim_graph",
            description="A claim-graph node that combines distinct blocker conditions under a union-null gate.",
            source_key="claim_graph",
        ),
        node(
            "arbitrary_dependence",
            "Arbitrary dependence",
            category="dependence_regime",
            domain="claim_graph",
            description="Dependence label used when child streams may be dependent in unknown ways.",
            source_key="claim_graph",
        ),
        node(
            "independent_dependence",
            "Independent dependence",
            category="dependence_regime",
            domain="claim_graph",
            description="Dependence label used when child streams are independent under the null.",
            source_key="claim_graph",
        ),
        node(
            "sequential_dependence",
            "Sequential dependence",
            category="dependence_regime",
            domain="claim_graph",
            description="Dependence label used when child factors form one sequential evidence stream.",
            source_key="claim_graph",
        ),
        node(
            "weighted_mean_merge",
            "Weighted mean merge",
            category="merge_operator",
            domain="claim_graph",
            description="Default valid merge for same-null evidence under arbitrary dependence.",
            source_key="claim_graph",
        ),
        node(
            "product_merge",
            "Sequential product merge",
            category="merge_operator",
            domain="claim_graph",
            description="Valid merge under sequential or independent null-side factorization assumptions.",
            source_key="claim_graph",
        ),
        node(
            "minimum_gate",
            "Minimum gate",
            category="merge_operator",
            domain="claim_graph",
            description="Union-null gate operator taking the minimum child e-process.",
            source_key="claim_graph",
        ),
    )

    edges = (
        edge("process_graph_prefix", "partitions_nodes_into", "object_type_family", domain="primitive", source_key="primitive"),
        edge("process_graph_prefix", "partitions_nodes_into", "event_type_family", domain="primitive", source_key="primitive"),
        edge("process_graph_prefix", "partitions_nodes_into", "context_type_family", domain="primitive", source_key="primitive"),
        edge("process_graph_prefix", "uses_edge_type", "involves_edge", domain="primitive", source_key="primitive"),
        edge("involves_edge", "connects_source_class", "event_type_family", domain="primitive", source_key="primitive"),
        edge("involves_edge", "connects_target_class", "object_type_family", domain="primitive", source_key="primitive"),
        edge("object_lifecycle", "defined_from", "involves_edge", domain="primitive", source_key="primitive"),
        edge("activity_trace", "ordered_version_of", "object_lifecycle", domain="primitive", source_key="primitive"),
        edge("milestone_time", "computed_from", "object_lifecycle", domain="primitive", source_key="primitive"),
        edge("duration_metric", "difference_of", "milestone_time", domain="primitive", source_key="primitive"),
        edge("oscar_agent", "operates_on", "process_graph_prefix", domain="agents", source_key="agents"),
        edge("kurt_agent", "operates_on", "process_graph_prefix", domain="agents", source_key="agents"),
        edge("global_model", "consumes_summaries_from", "oscar_agent", domain="agents", source_key="agents"),
        edge("global_model", "consumes_summaries_from", "kurt_agent", domain="agents", source_key="agents"),
        edge("global_model", "recommends", "intervention_templates", domain="agents", source_key="agents"),
        edge("global_model", "recommends", "monitor_templates", domain="agents", source_key="agents"),
        edge("typed_temporal_context", "reified_as", "higher_layer_object", domain="recursive", source_key="recursive"),
        edge("typed_temporal_context", "reified_as", "higher_layer_event", domain="recursive", source_key="recursive"),
        edge("hierarchical_graph", "lifts", "typed_temporal_context", domain="recursive", source_key="recursive"),
        edge("hierarchical_graph", "contains", "higher_layer_object", domain="recursive", source_key="recursive"),
        edge("hierarchical_graph", "contains", "higher_layer_event", domain="recursive", source_key="recursive"),
        edge("approval_claim_graph", "contains_node_class", "leaf_evidence_node", domain="claim_graph", source_key="claim_graph"),
        edge("approval_claim_graph", "contains_node_class", "same_null_merge_node", domain="claim_graph", source_key="claim_graph"),
        edge("approval_claim_graph", "contains_node_class", "gate_node", domain="claim_graph", source_key="claim_graph"),
        edge("same_null_merge_node", "tracks_dependence_regime", "arbitrary_dependence", domain="claim_graph", source_key="claim_graph"),
        edge("same_null_merge_node", "tracks_dependence_regime", "independent_dependence", domain="claim_graph", source_key="claim_graph"),
        edge("same_null_merge_node", "tracks_dependence_regime", "sequential_dependence", domain="claim_graph", source_key="claim_graph"),
        edge("same_null_merge_node", "uses_merge_operator", "weighted_mean_merge", domain="claim_graph", source_key="claim_graph"),
        edge("same_null_merge_node", "uses_merge_operator", "product_merge", domain="claim_graph", source_key="claim_graph"),
        edge("gate_node", "uses_merge_operator", "minimum_gate", domain="claim_graph", source_key="claim_graph"),
        edge("minimum_gate", "applies_to", "gate_node", domain="claim_graph", source_key="claim_graph"),
    )
    return nodes, edges


def _nodes_by_id(nodes: Sequence[OscarGraphNode]) -> dict[str, OscarGraphNode]:
    return {node.node_id: node for node in nodes}


def _serialize_task(
    *,
    task_id: str,
    family: str,
    difficulty: int,
    question: str,
    context_payload: dict[str, object],
    target_answer: object,
    auxiliary_targets: dict[str, object],
) -> str:
    state_tokens = (
        f"family={family}",
        f"graph.domain={auxiliary_targets.get('graph_domain', 'mixed')}",
        f"graph.source_category={auxiliary_targets.get('source_category', 'unknown')}",
        f"graph.target_category={auxiliary_targets.get('target_category', 'unknown')}",
    )
    lines = [
        FORMAT_HEADER,
        "record_type=terminal_answer",
        f"dataset={OSCAR_GRAPH_REASONING_BENCHMARK}",
        "source_modality=process_graph_schema",
        f"family={family}",
        f"difficulty={difficulty}",
        f"trajectory_id={task_id}",
        "step_index=0",
        "trace_length=1",
        "trace_step=reason",
        "previous_action=<start>",
        f"candidate_bucket=family:{family}|step:0|prev:<start>",
        "answer_format=json",
        "variant_kind=domain_graph_reasoning",
        "trajectory_role=gold",
        "state_tokens=" + " ".join(state_tokens),
        "state_scalars=0.000000,0.000000,0.000000",
        "question=" + json.dumps(question, separators=(",", ":"), sort_keys=True),
        "context=" + json.dumps(context_payload, separators=(",", ":"), sort_keys=True),
        "auxiliary_targets=" + json.dumps(auxiliary_targets, separators=(",", ":"), sort_keys=True),
        "target_answer=" + json.dumps(target_answer, separators=(",", ":"), sort_keys=True),
    ]
    return "\n".join(lines) + "\n"


def _serialize_executor_decision_step(
    *,
    trajectory_id: str,
    family: str,
    difficulty: int,
    step_index: int,
    trace_length: int,
    trace_step: str,
    previous_action: str,
    candidate_bucket: str,
    question: str,
    context_payload: dict[str, object],
    verifier_context: dict[str, object],
    state_tokens: Sequence[str],
    state_scalars: Sequence[float],
    auxiliary_targets: dict[str, object],
    target_action: dict[str, object],
) -> str:
    lines = [
        FORMAT_HEADER,
        "record_type=decision_action",
        f"dataset={OSCAR_GRAPH_REASONING_BENCHMARK}",
        "source_modality=process_graph_schema",
        f"family={family}",
        f"difficulty={difficulty}",
        f"trajectory_id={trajectory_id}",
        f"step_index={step_index}",
        f"trace_length={trace_length}",
        f"trace_step={trace_step}",
        f"previous_action={previous_action}",
        f"candidate_bucket={candidate_bucket}",
        "answer_format=json",
        "variant_kind=domain_graph_executor",
        "trajectory_role=gold",
        "state_tokens=" + " ".join(state_tokens),
        "state_scalars=" + ",".join(f"{value:.6f}" for value in state_scalars),
        "verifier_context=" + json.dumps(verifier_context, separators=(",", ":"), sort_keys=True),
        "question=" + json.dumps(question, separators=(",", ":"), sort_keys=True),
        "context=" + json.dumps(context_payload, separators=(",", ":"), sort_keys=True),
        "auxiliary_targets=" + json.dumps(auxiliary_targets, separators=(",", ":"), sort_keys=True),
        "target_action=" + json.dumps(target_action, separators=(",", ":"), sort_keys=True),
    ]
    return "\n".join(lines) + "\n"


def _subgraph_payload(
    *,
    nodes_by_id: dict[str, OscarGraphNode],
    edges: Sequence[OscarGraphEdge],
    focus_node_ids: Iterable[str],
) -> dict[str, object]:
    focus = {node_id for node_id in focus_node_ids if node_id in nodes_by_id}
    local_edges = [
        edge
        for edge in edges
        if edge.source_id in focus or edge.target_id in focus
    ]
    expanded_focus = focus.union({edge.source_id for edge in local_edges}).union({edge.target_id for edge in local_edges})
    local_nodes = [nodes_by_id[node_id] for node_id in sorted(expanded_focus) if node_id in nodes_by_id]
    return {
        "nodes": [
            {
                "id": node.node_id,
                "label": node.label,
                "category": node.category,
                "domain": node.domain,
            }
            for node in local_nodes
        ],
        "edges": [
            {
                "source": edge.source_id,
                "relation": edge.relation,
                "target": edge.target_id,
                "domain": edge.domain,
            }
            for edge in local_edges
        ],
    }


def _split_scope_text(record_text: str) -> tuple[dict[str, str], str]:
    if "\n\n" in record_text:
        header_block, body = record_text.split("\n\n", 1)
    else:
        header_block, body = record_text, ""
    header_fields: dict[str, str] = {}
    for raw_line in header_block.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        header_fields[key] = value
    return header_fields, body.strip()


def _grounding_score(node_id: str, text: str) -> int:
    lowered = f" {text.lower()} "
    score = 0
    for keyword in _GROUNDING_KEYWORDS.get(node_id, ()):
        if keyword.lower() in lowered:
            score += 1
    return score


def _grounding_candidate_nodes(nodes: Sequence[OscarGraphNode], text: str) -> list[tuple[OscarGraphNode, int]]:
    scored = [(node, _grounding_score(node.node_id, text)) for node in nodes]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: (-item[1], item[0].domain, item[0].label.lower()))
    return scored


def _executor_motifs() -> tuple[OscarGraphExecutorMotif, ...]:
    return (
        OscarGraphExecutorMotif(
            motif_id="primitive_object_binding",
            motif_label="Primitive object binding",
            graph_domain="primitive",
            abstraction_level="schema_binding",
            source_node_id="process_graph_prefix",
            rollout_node_ids=("involves_edge", "object_type_family"),
            rollout_relations=("uses_edge_type", "connects_target_class"),
            cue_keywords=("persistent objects", "object-centric", "involves"),
        ),
        OscarGraphExecutorMotif(
            motif_id="primitive_event_binding",
            motif_label="Primitive event binding",
            graph_domain="primitive",
            abstraction_level="schema_binding",
            source_node_id="process_graph_prefix",
            rollout_node_ids=("involves_edge", "event_type_family"),
            rollout_relations=("uses_edge_type", "connects_source_class"),
            cue_keywords=("timestamped activities", "event types", "involves"),
        ),
        OscarGraphExecutorMotif(
            motif_id="agent_control_surface",
            motif_label="Agent control surface",
            graph_domain="agents",
            abstraction_level="agentic_control",
            source_node_id="global_model",
            rollout_node_ids=("oscar_agent", "process_graph_prefix"),
            rollout_relations=("consumes_summaries_from", "operates_on"),
            cue_keywords=("oscar state", "policy control", "practice compiler"),
        ),
        OscarGraphExecutorMotif(
            motif_id="agent_observability_surface",
            motif_label="Agent observability surface",
            graph_domain="agents",
            abstraction_level="agentic_observability",
            source_node_id="global_model",
            rollout_node_ids=("kurt_agent", "process_graph_prefix"),
            rollout_relations=("consumes_summaries_from", "operates_on"),
            cue_keywords=("observability", "kurt state", "model-training"),
        ),
        OscarGraphExecutorMotif(
            motif_id="recursive_object_reification",
            motif_label="Recursive object reification",
            graph_domain="recursive",
            abstraction_level="recursive_abstraction",
            source_node_id="hierarchical_graph",
            rollout_node_ids=("typed_temporal_context", "higher_layer_object"),
            rollout_relations=("lifts", "reified_as"),
            cue_keywords=("higher-layer object", "reified", "typed temporal context"),
        ),
        OscarGraphExecutorMotif(
            motif_id="recursive_event_reification",
            motif_label="Recursive event reification",
            graph_domain="recursive",
            abstraction_level="recursive_abstraction",
            source_node_id="hierarchical_graph",
            rollout_node_ids=("typed_temporal_context", "higher_layer_event"),
            rollout_relations=("lifts", "reified_as"),
            cue_keywords=("higher-layer event", "reified", "typed temporal context"),
        ),
        OscarGraphExecutorMotif(
            motif_id="claim_arbitrary_merge",
            motif_label="Claim arbitrary merge",
            graph_domain="claim_graph",
            abstraction_level="claim_aggregation",
            source_node_id="approval_claim_graph",
            rollout_node_ids=("same_null_merge_node", "weighted_mean_merge"),
            rollout_relations=("contains_node_class", "uses_merge_operator"),
            cue_keywords=("same-null", "weighted mean merge", "arbitrary dependence"),
        ),
        OscarGraphExecutorMotif(
            motif_id="claim_factorized_merge",
            motif_label="Claim factorized merge",
            graph_domain="claim_graph",
            abstraction_level="claim_aggregation",
            source_node_id="approval_claim_graph",
            rollout_node_ids=("same_null_merge_node", "product_merge"),
            rollout_relations=("contains_node_class", "uses_merge_operator"),
            cue_keywords=("product process", "sequential", "independent under the null"),
        ),
        OscarGraphExecutorMotif(
            motif_id="claim_union_gate",
            motif_label="Claim union gate",
            graph_domain="claim_graph",
            abstraction_level="claim_gating",
            source_node_id="approval_claim_graph",
            rollout_node_ids=("gate_node", "minimum_gate"),
            rollout_relations=("contains_node_class", "uses_merge_operator"),
            cue_keywords=("gate nodes", "distinct blockers", "minimum gate"),
        ),
    )


def _best_executor_support_record(
    motif: OscarGraphExecutorMotif,
    records: Sequence[OscarScopeRecord],
) -> OscarScopeRecord | None:
    best_record = None
    best_score = 0
    for record in records:
        if record.view != "native_chunk":
            continue
        _header, body = _split_scope_text(record.text)
        lowered = body.lower()
        score = sum(1 for keyword in motif.cue_keywords if keyword.lower() in lowered)
        if score > best_score:
            best_record = record
            best_score = score
    return best_record if best_score > 0 else None


def _relation_tasks(nodes: Sequence[OscarGraphNode], edges: Sequence[OscarGraphEdge]) -> list[OscarGraphReasoningTask]:
    nodes_by_id = _nodes_by_id(nodes)
    relation_candidates = sorted({edge.relation for edge in edges})
    tasks: list[OscarGraphReasoningTask] = []
    for edge_index, schema_edge in enumerate(edges):
        source = nodes_by_id[schema_edge.source_id]
        target = nodes_by_id[schema_edge.target_id]
        question = f"What relation connects {source.label} to {target.label} in the canonical Oscar graph?"
        auxiliary_targets = {
            "family": "oscar_graph_relation",
            "task_kind": "graph_relation",
            "graph_domain": schema_edge.domain,
            "source_node_id": source.node_id,
            "source_label": source.label,
            "source_category": source.category,
            "target_node_id": target.node_id,
            "target_label": target.label,
            "target_category": target.category,
            "relation": schema_edge.relation,
            "relation_candidates": relation_candidates,
            "provenance": list(schema_edge.provenance),
        }
        context = {
            "source": {"label": source.label, "category": source.category, "description": source.description},
            "target": {"label": target.label, "category": target.category, "description": target.description},
            "local_graph": _subgraph_payload(nodes_by_id=nodes_by_id, edges=edges, focus_node_ids=(source.node_id, target.node_id)),
        }
        task_id = f"oscar_graph_relation:{edge_index}"
        text = _serialize_task(
            task_id=task_id,
            family="oscar_graph_relation",
            difficulty=2,
            question=question,
            context_payload=context,
            target_answer={"relation": schema_edge.relation},
            auxiliary_targets=auxiliary_targets,
        )
        tasks.append(
            OscarGraphReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_GRAPH_REASONING_BENCHMARK,
                family="oscar_graph_relation",
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return tasks


def _neighbor_tasks(nodes: Sequence[OscarGraphNode], edges: Sequence[OscarGraphEdge]) -> list[OscarGraphReasoningTask]:
    nodes_by_id = _nodes_by_id(nodes)
    grouped: dict[tuple[str, str], list[OscarGraphEdge]] = {}
    for schema_edge in edges:
        grouped.setdefault((schema_edge.source_id, schema_edge.relation), []).append(schema_edge)
    tasks: list[OscarGraphReasoningTask] = []
    for task_index, ((source_id, relation), grouped_edges) in enumerate(sorted(grouped.items())):
        source = nodes_by_id[source_id]
        targets = [nodes_by_id[edge.target_id] for edge in grouped_edges]
        question = f"Which direct targets does {source.label} connect to by relation {relation}?"
        auxiliary_targets = {
            "family": "oscar_graph_neighbors",
            "task_kind": "graph_neighbors",
            "graph_domain": grouped_edges[0].domain,
            "source_node_id": source.node_id,
            "source_label": source.label,
            "source_category": source.category,
            "relation": relation,
            "target_category": targets[0].category if targets else "unknown",
            "candidate_target_ids": [node.node_id for node in nodes],
            "candidate_targets": [node.label for node in nodes],
            "target_node_ids": [node.node_id for node in targets],
            "target_labels": [node.label for node in targets],
            "provenance": list(grouped_edges[0].provenance),
        }
        context = {
            "source": {"label": source.label, "category": source.category, "description": source.description},
            "relation": relation,
            "local_graph": _subgraph_payload(
                nodes_by_id=nodes_by_id,
                edges=edges,
                focus_node_ids=(source.node_id, *(node.node_id for node in targets)),
            ),
        }
        task_id = f"oscar_graph_neighbors:{task_index}"
        text = _serialize_task(
            task_id=task_id,
            family="oscar_graph_neighbors",
            difficulty=3,
            question=question,
            context_payload=context,
            target_answer={"targets": [node.label for node in targets]},
            auxiliary_targets=auxiliary_targets,
        )
        tasks.append(
            OscarGraphReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_GRAPH_REASONING_BENCHMARK,
                family="oscar_graph_neighbors",
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return tasks


def _path_completion_tasks(nodes: Sequence[OscarGraphNode], edges: Sequence[OscarGraphEdge]) -> list[OscarGraphReasoningTask]:
    nodes_by_id = _nodes_by_id(nodes)
    outgoing: dict[str, list[OscarGraphEdge]] = {}
    for schema_edge in edges:
        outgoing.setdefault(schema_edge.source_id, []).append(schema_edge)
    tasks: list[OscarGraphReasoningTask] = []
    task_index = 0
    for first_edge in edges:
        for second_edge in outgoing.get(first_edge.target_id, ()):
            source = nodes_by_id[first_edge.source_id]
            via = nodes_by_id[first_edge.target_id]
            target = nodes_by_id[second_edge.target_id]
            question = (
                f"Complete the two-hop path from {source.label}: "
                f"{first_edge.relation} then {second_edge.relation}. Which intermediate and final nodes appear?"
            )
            auxiliary_targets = {
                "family": "oscar_graph_path_completion",
                "task_kind": "graph_path_completion",
                "graph_domain": first_edge.domain,
                "source_node_id": source.node_id,
                "source_label": source.label,
                "source_category": source.category,
                "via_node_id": via.node_id,
                "via_label": via.label,
                "via_category": via.category,
                "target_node_id": target.node_id,
                "target_label": target.label,
                "target_category": target.category,
                "first_relation": first_edge.relation,
                "second_relation": second_edge.relation,
                "candidate_node_ids": [node.node_id for node in nodes],
                "candidate_nodes": [node.label for node in nodes],
                "provenance": list(first_edge.provenance),
            }
            context = {
                "source": {"label": source.label, "category": source.category, "description": source.description},
                "path_relations": [first_edge.relation, second_edge.relation],
                "local_graph": _subgraph_payload(
                    nodes_by_id=nodes_by_id,
                    edges=edges,
                    focus_node_ids=(source.node_id, via.node_id, target.node_id),
                ),
            }
            task_id = f"oscar_graph_path_completion:{task_index}"
            task_index += 1
            text = _serialize_task(
                task_id=task_id,
                family="oscar_graph_path_completion",
                difficulty=4,
                question=question,
                context_payload=context,
                target_answer={"via": via.label, "target": target.label},
                auxiliary_targets=auxiliary_targets,
            )
            tasks.append(
                OscarGraphReasoningTask(
                    task_id=task_id,
                    benchmark=OSCAR_GRAPH_REASONING_BENCHMARK,
                    family="oscar_graph_path_completion",
                    trace_step="reason",
                    text=text,
                    metadata=auxiliary_targets,
                )
            )
    return tasks


def _grounding_tasks(
    nodes: Sequence[OscarGraphNode],
    edges: Sequence[OscarGraphEdge],
) -> list[OscarGraphReasoningTask]:
    records = build_oscar_scope_records(
        auto_discover=True,
        views=("native_chunk",),
        max_documents=8,
        max_chunks=96,
    )
    nodes_by_id = _nodes_by_id(nodes)
    tasks: list[OscarGraphReasoningTask] = []
    task_index = 0
    candidate_labels = [node.label for node in nodes]
    for record in records:
        if record.view != "native_chunk":
            continue
        header, body = _split_scope_text(record.text)
        section_path = str(record.metadata.get("section_path", header.get("section_path", record.document_id)))
        doc_title = str(record.metadata.get("doc_title", header.get("doc_title", record.document_id)))
        scored_candidates = _grounding_candidate_nodes(nodes, " ".join((doc_title, section_path, body)))
        if not scored_candidates:
            continue
        best_node, best_score = scored_candidates[0]
        if best_score <= 0:
            continue
        top_candidates = [node for node, _score in scored_candidates[:6]]
        question = "Which canonical Oscar graph node best abstracts this process excerpt?"
        auxiliary_targets = {
            "family": "oscar_graph_grounding",
            "task_kind": "graph_grounding",
            "graph_domain": best_node.domain,
            "source_category": "process_excerpt",
            "target_node_id": best_node.node_id,
            "target_label": best_node.label,
            "target_category": best_node.category,
            "doc_id": record.document_id,
            "doc_title": doc_title,
            "section_path": section_path,
            "candidate_node_ids": [node.node_id for node in top_candidates],
            "candidate_nodes": [node.label for node in top_candidates],
            "provenance": list(best_node.provenance),
        }
        context = {
            "excerpt": body,
            "doc_title": doc_title,
            "section_path": section_path,
            "candidate_nodes": [
                {
                    "label": node.label,
                    "category": node.category,
                    "domain": node.domain,
                    "description": node.description,
                }
                for node in top_candidates
            ],
            "local_graph": _subgraph_payload(nodes_by_id=nodes_by_id, edges=edges, focus_node_ids=(best_node.node_id,)),
        }
        task_id = f"oscar_graph_grounding:{task_index}"
        task_index += 1
        text = _serialize_task(
            task_id=task_id,
            family="oscar_graph_grounding",
            difficulty=min(5, 2 + best_score),
            question=question,
            context_payload=context,
            target_answer={"node": best_node.label},
            auxiliary_targets=auxiliary_targets,
        )
        tasks.append(
            OscarGraphReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_GRAPH_REASONING_BENCHMARK,
                family="oscar_graph_grounding",
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return tasks


def _executor_rollout_tasks(
    nodes: Sequence[OscarGraphNode],
    edges: Sequence[OscarGraphEdge],
) -> list[OscarGraphReasoningTask]:
    nodes_by_id = _nodes_by_id(nodes)
    records = build_oscar_scope_records(
        auto_discover=True,
        views=("native_chunk",),
        max_documents=8,
        max_chunks=96,
    )
    tasks: list[OscarGraphReasoningTask] = []
    candidate_node_ids = [node.node_id for node in nodes]
    candidate_nodes = [node.label for node in nodes]
    motif_candidates = [motif.motif_label for motif in _executor_motifs()]
    for task_index, motif in enumerate(_executor_motifs()):
        source = nodes_by_id[motif.source_node_id]
        rollout_nodes = [nodes_by_id[node_id] for node_id in motif.rollout_node_ids]
        support_record = _best_executor_support_record(motif, records)
        support_excerpt = ""
        support_doc_title = ""
        support_section_path = ""
        if support_record is not None:
            header, body = _split_scope_text(support_record.text)
            support_excerpt = body
            support_doc_title = str(
                support_record.metadata.get("doc_title", header.get("doc_title", support_record.document_id))
            )
            support_section_path = str(
                support_record.metadata.get("section_path", header.get("section_path", support_record.document_id))
            )
        question = (
            f"Execute the canonical Oscar graph rollout for motif {motif.motif_label} starting from {source.label}. "
            "Which nodes should the graph executor visit at each step, and which abstraction motif is being instantiated?"
        )
        auxiliary_targets = {
            "family": "oscar_graph_executor_rollout",
            "task_kind": "graph_executor_rollout",
            "graph_domain": motif.graph_domain,
            "source_node_id": source.node_id,
            "source_label": source.label,
            "source_category": source.category,
            "target_node_id": rollout_nodes[-1].node_id,
            "target_label": rollout_nodes[-1].label,
            "target_category": rollout_nodes[-1].category,
            "rollout_step_node_ids": [node.node_id for node in rollout_nodes],
            "rollout_step_labels": [node.label for node in rollout_nodes],
            "rollout_step_relations": list(motif.rollout_relations),
            "rollout_step_count": len(rollout_nodes),
            "rollout_candidate_node_ids": candidate_node_ids,
            "rollout_candidate_nodes": candidate_nodes,
            "rollout_motif_id": motif.motif_id,
            "rollout_motif_label": motif.motif_label,
            "rollout_motif_candidates": motif_candidates,
            "abstraction_level": motif.abstraction_level,
            "support_excerpt": support_excerpt,
            "support_doc_title": support_doc_title,
            "support_section_path": support_section_path,
            "provenance": list(source.provenance),
        }
        context = {
            "source": {
                "label": source.label,
                "category": source.category,
                "description": source.description,
            },
            "rollout_relations": list(motif.rollout_relations),
            "motif_label": motif.motif_label,
            "abstraction_level": motif.abstraction_level,
            "support_excerpt": support_excerpt,
            "support_doc_title": support_doc_title,
            "support_section_path": support_section_path,
            "local_graph": _subgraph_payload(
                nodes_by_id=nodes_by_id,
                edges=edges,
                focus_node_ids=(source.node_id, *(node.node_id for node in rollout_nodes)),
            ),
        }
        task_id = f"oscar_graph_executor_rollout:{task_index}"
        text = _serialize_task(
            task_id=task_id,
            family="oscar_graph_executor_rollout",
            difficulty=5,
            question=question,
            context_payload=context,
            target_answer={
                "rollout": [node.label for node in rollout_nodes],
                "motif": motif.motif_label,
            },
            auxiliary_targets=auxiliary_targets,
        )
        tasks.append(
            OscarGraphReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_GRAPH_REASONING_BENCHMARK,
                family="oscar_graph_executor_rollout",
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return tasks


def _executor_trace_tasks(
    nodes: Sequence[OscarGraphNode],
    edges: Sequence[OscarGraphEdge],
) -> list[OscarGraphReasoningTask]:
    nodes_by_id = _nodes_by_id(nodes)
    motif_by_id = {motif.motif_id: motif for motif in _executor_motifs()}
    records = build_oscar_scope_records(
        auto_discover=True,
        views=("native_chunk",),
        max_documents=8,
        max_chunks=96,
    )
    motif_candidates = [motif.motif_label for motif in _executor_motifs()]
    claim_root_targets = (
        "leaf_evidence_node",
        "same_null_merge_node",
        "gate_node",
    )
    tasks: list[OscarGraphReasoningTask] = []

    def candidate_bucket(*, motif_id: str, step_index: int, previous_action_name: str, branch_slot: str) -> str:
        return (
            f"family:oscar_graph_executor_trace|motif:{motif_id}|step:{step_index}|"
            f"prev:{previous_action_name}|branch:{branch_slot}"
        )

    def add_plan(
        *,
        motif_id: str,
        branch_steps: Sequence[tuple[str, str, str, Sequence[str], str]],
    ) -> None:
        motif = motif_by_id[motif_id]
        source = nodes_by_id[motif.source_node_id]
        support_record = _best_executor_support_record(motif, records)
        support_excerpt = ""
        support_doc_title = ""
        support_section_path = ""
        if support_record is not None:
            header, body = _split_scope_text(support_record.text)
            support_excerpt = body
            support_doc_title = str(
                support_record.metadata.get("doc_title", header.get("doc_title", support_record.document_id))
            )
            support_section_path = str(
                support_record.metadata.get("section_path", header.get("section_path", support_record.document_id))
            )
        total_steps = 1 + len(branch_steps)
        trajectory_id = f"oscar_graph_executor_trace:{motif_id}"
        previous_action = "<start>"
        previous_action_name = "<start>"
        visited_node_ids = [source.node_id]

        target_action = {
            "name": "select_motif",
            "action": {
                "motif_id": motif.motif_id,
                "motif_label": motif.motif_label,
            },
        }
        auxiliary_targets = {
            "family": "oscar_graph_executor_trace",
            "task_kind": "graph_executor_step",
            "graph_domain": motif.graph_domain,
            "source_node_id": source.node_id,
            "source_label": source.label,
            "source_category": source.category,
            "rollout_motif_id": motif.motif_id,
            "rollout_motif_label": motif.motif_label,
            "rollout_motif_candidates": motif_candidates,
            "abstraction_level": motif.abstraction_level,
            "support_excerpt": support_excerpt,
            "support_doc_title": support_doc_title,
            "support_section_path": support_section_path,
            "candidate_node_ids": [],
            "rollout_candidate_node_ids": [],
            "rollout_step_node_ids": [],
            "rollout_step_relations": [],
            "pending_branches": [
                {
                    "branch_slot": slot,
                    "current_node_id": current_node_id,
                    "relation": relation,
                    "target_node_id": target_node_id,
                }
                for slot, current_node_id, relation, _candidates, target_node_id in branch_steps
            ],
            "visited_node_ids": visited_node_ids,
            "target_action_name": "select_motif",
            "provenance": list(source.provenance),
            "trajectory_id": trajectory_id,
            "step_index": 0,
            "trace_length": total_steps,
        }
        context = {
            "source": {
                "label": source.label,
                "category": source.category,
                "description": source.description,
            },
            "motif_candidates": motif_candidates,
            "support_excerpt": support_excerpt,
            "support_doc_title": support_doc_title,
            "support_section_path": support_section_path,
            "local_graph": _subgraph_payload(
                nodes_by_id=nodes_by_id,
                edges=edges,
                focus_node_ids=(source.node_id, *(target_node_id for *_prefix, target_node_id in branch_steps)),
            ),
        }
        text = _serialize_executor_decision_step(
            trajectory_id=trajectory_id,
            family="oscar_graph_executor_trace",
            difficulty=5,
            step_index=0,
            trace_length=total_steps,
            trace_step="select_motif",
            previous_action=previous_action,
            candidate_bucket=candidate_bucket(
                motif_id=motif.motif_id,
                step_index=0,
                previous_action_name=previous_action_name,
                branch_slot="motif",
            ),
            question="Which canonical executor motif best matches this Oscar graph rollout context?",
            context_payload=context,
            verifier_context={
                "next_branch_slot": branch_steps[0][0] if branch_steps else None,
                "pending_branch_count": len(branch_steps),
                "resolved_branch_count": 0,
                "should_stop": False,
            },
            state_tokens=(
                f"family=oscar_graph_executor_trace",
                f"graph.domain={motif.graph_domain}",
                f"executor.current={source.node_id}",
                f"executor.mode=branching",
                f"executor.pending={len(branch_steps)}",
                f"abstraction.level={motif.abstraction_level}",
            ),
            state_scalars=(0.0, 0.0, float(len(branch_steps))),
            auxiliary_targets=auxiliary_targets,
            target_action=target_action,
        )
        tasks.append(
            OscarGraphReasoningTask(
                task_id=f"{trajectory_id}:step_0",
                benchmark=OSCAR_GRAPH_REASONING_BENCHMARK,
                family="oscar_graph_executor_trace",
                trace_step="select_motif",
                text=text,
                metadata=auxiliary_targets,
            )
        )
        previous_action = json.dumps(target_action, separators=(",", ":"), sort_keys=True)
        previous_action_name = "select_motif"

        for step_offset, (branch_slot, current_node_id, relation, candidates, target_node_id) in enumerate(branch_steps, start=1):
            target_action = {
                "name": "advance_frontier",
                "action": {
                    "branch_slot": branch_slot,
                    "relation": relation,
                    "selected_node_id": target_node_id,
                },
            }
            remaining_branches = branch_steps[step_offset:]
            target_node = nodes_by_id[target_node_id]
            auxiliary_targets = {
                "family": "oscar_graph_executor_trace",
                "task_kind": "graph_executor_step",
                "graph_domain": motif.graph_domain,
                "source_node_id": current_node_id,
                "source_label": nodes_by_id[current_node_id].label,
                "source_category": nodes_by_id[current_node_id].category,
                "target_node_id": target_node_id,
                "target_label": target_node.label,
                "target_category": target_node.category,
                "relation": relation,
                "rollout_motif_id": motif.motif_id,
                "rollout_motif_label": motif.motif_label,
                "rollout_motif_candidates": motif_candidates,
                "abstraction_level": motif.abstraction_level,
                "branch_slot": branch_slot,
                "support_excerpt": support_excerpt,
                "support_doc_title": support_doc_title,
                "support_section_path": support_section_path,
                "candidate_node_ids": list(candidates),
                "candidate_nodes": [nodes_by_id[node_id].label for node_id in candidates],
                "rollout_candidate_node_ids": list(candidates),
                "rollout_step_node_ids": [target_node_id],
                "rollout_step_relations": [relation],
                "pending_branches": [
                    {
                        "branch_slot": next_slot,
                        "current_node_id": next_current_node_id,
                        "relation": next_relation,
                        "target_node_id": next_target_node_id,
                    }
                    for next_slot, next_current_node_id, next_relation, _next_candidates, next_target_node_id in remaining_branches
                ],
                "visited_node_ids": visited_node_ids,
                "target_action_name": "advance_frontier",
                "provenance": list(target_node.provenance),
                "trajectory_id": trajectory_id,
                "step_index": step_offset,
                "trace_length": total_steps,
            }
            context = {
                "current_node": {
                    "id": current_node_id,
                    "label": nodes_by_id[current_node_id].label,
                    "category": nodes_by_id[current_node_id].category,
                },
                "relation": relation,
                "branch_slot": branch_slot,
                "candidate_nodes": [
                    {
                        "id": node_id,
                        "label": nodes_by_id[node_id].label,
                        "category": nodes_by_id[node_id].category,
                        "domain": nodes_by_id[node_id].domain,
                    }
                    for node_id in candidates
                ],
                "visited_node_ids": visited_node_ids,
                "pending_branches": [
                    {
                        "branch_slot": next_slot,
                        "current_node_id": next_current_node_id,
                        "relation": next_relation,
                        "target_node_id": next_target_node_id,
                    }
                    for next_slot, next_current_node_id, next_relation, _next_candidates, next_target_node_id in remaining_branches
                ],
                "support_excerpt": support_excerpt,
                "local_graph": _subgraph_payload(
                    nodes_by_id=nodes_by_id,
                    edges=edges,
                    focus_node_ids=(current_node_id, *candidates),
                ),
            }
            text = _serialize_executor_decision_step(
                trajectory_id=trajectory_id,
                family="oscar_graph_executor_trace",
                difficulty=5,
                step_index=step_offset,
                trace_length=total_steps,
                trace_step="advance_frontier",
                previous_action=previous_action,
                candidate_bucket=candidate_bucket(
                    motif_id=motif.motif_id,
                    step_index=step_offset,
                    previous_action_name=previous_action_name,
                    branch_slot=branch_slot,
                ),
                question=f"Advance branch {branch_slot} from {nodes_by_id[current_node_id].label} via relation {relation}.",
                context_payload=context,
                verifier_context={
                    "next_branch_slot": remaining_branches[0][0] if remaining_branches else None,
                    "pending_branch_count": len(remaining_branches),
                    "resolved_branch_count": step_offset,
                    "should_stop": step_offset + 1 >= total_steps,
                },
                state_tokens=(
                    f"family=oscar_graph_executor_trace",
                    f"graph.domain={motif.graph_domain}",
                    f"executor.current={current_node_id}",
                    f"executor.branch_slot={branch_slot}",
                    f"executor.relation={relation}",
                    f"executor.pending={len(remaining_branches)}",
                    f"executor.visited={len(visited_node_ids)}",
                    f"motif={motif.motif_id}",
                ),
                state_scalars=(
                    step_offset / float(max(total_steps - 1, 1)),
                    float(len(visited_node_ids)),
                    float(len(remaining_branches)),
                ),
                auxiliary_targets=auxiliary_targets,
                target_action=target_action,
            )
            tasks.append(
                OscarGraphReasoningTask(
                    task_id=f"{trajectory_id}:step_{step_offset}",
                    benchmark=OSCAR_GRAPH_REASONING_BENCHMARK,
                    family="oscar_graph_executor_trace",
                    trace_step="advance_frontier",
                    text=text,
                    metadata=auxiliary_targets,
                )
            )
            previous_action = json.dumps(target_action, separators=(",", ":"), sort_keys=True)
            previous_action_name = "advance_frontier"
            visited_node_ids = visited_node_ids + [target_node_id]

    add_plan(
        motif_id="claim_arbitrary_merge",
        branch_steps=(
            ("root", "approval_claim_graph", "contains_node_class", claim_root_targets, "same_null_merge_node"),
            (
                "dependence",
                "same_null_merge_node",
                "tracks_dependence_regime",
                ("arbitrary_dependence", "independent_dependence", "sequential_dependence"),
                "arbitrary_dependence",
            ),
            (
                "merge",
                "same_null_merge_node",
                "uses_merge_operator",
                ("weighted_mean_merge", "product_merge"),
                "weighted_mean_merge",
            ),
        ),
    )
    add_plan(
        motif_id="claim_factorized_merge",
        branch_steps=(
            ("root", "approval_claim_graph", "contains_node_class", claim_root_targets, "same_null_merge_node"),
            (
                "dependence",
                "same_null_merge_node",
                "tracks_dependence_regime",
                ("arbitrary_dependence", "independent_dependence", "sequential_dependence"),
                "sequential_dependence",
            ),
            (
                "merge",
                "same_null_merge_node",
                "uses_merge_operator",
                ("weighted_mean_merge", "product_merge"),
                "product_merge",
            ),
        ),
    )
    add_plan(
        motif_id="claim_union_gate",
        branch_steps=(
            ("root", "approval_claim_graph", "contains_node_class", claim_root_targets, "gate_node"),
            ("merge", "gate_node", "uses_merge_operator", ("minimum_gate",), "minimum_gate"),
        ),
    )
    add_plan(
        motif_id="recursive_object_reification",
        branch_steps=(
            ("lift", "hierarchical_graph", "lifts", ("typed_temporal_context",), "typed_temporal_context"),
            ("reify", "typed_temporal_context", "reified_as", ("higher_layer_object", "higher_layer_event"), "higher_layer_object"),
        ),
    )
    add_plan(
        motif_id="recursive_event_reification",
        branch_steps=(
            ("lift", "hierarchical_graph", "lifts", ("typed_temporal_context",), "typed_temporal_context"),
            ("reify", "typed_temporal_context", "reified_as", ("higher_layer_object", "higher_layer_event"), "higher_layer_event"),
        ),
    )
    add_plan(
        motif_id="agent_control_surface",
        branch_steps=(
            ("summary_source", "global_model", "consumes_summaries_from", ("oscar_agent", "kurt_agent"), "oscar_agent"),
            ("operate", "oscar_agent", "operates_on", ("process_graph_prefix",), "process_graph_prefix"),
        ),
    )
    add_plan(
        motif_id="agent_observability_surface",
        branch_steps=(
            ("summary_source", "global_model", "consumes_summaries_from", ("oscar_agent", "kurt_agent"), "kurt_agent"),
            ("operate", "kurt_agent", "operates_on", ("process_graph_prefix",), "process_graph_prefix"),
        ),
    )
    add_plan(
        motif_id="primitive_object_binding",
        branch_steps=(
            ("edge_type", "process_graph_prefix", "uses_edge_type", ("involves_edge",), "involves_edge"),
            ("binding_target", "involves_edge", "connects_target_class", ("object_type_family",), "object_type_family"),
        ),
    )
    add_plan(
        motif_id="primitive_event_binding",
        branch_steps=(
            ("edge_type", "process_graph_prefix", "uses_edge_type", ("involves_edge",), "involves_edge"),
            ("binding_source", "involves_edge", "connects_source_class", ("event_type_family",), "event_type_family"),
        ),
    )
    return tasks


def _round_robin_limit(
    task_groups: Sequence[Sequence[OscarGraphReasoningTask]],
    *,
    max_examples: int | None,
) -> list[OscarGraphReasoningTask]:
    ordered_groups = [list(group) for group in task_groups if group]
    if max_examples is None:
        return [task for group in ordered_groups for task in group]
    if max_examples <= 0 or not ordered_groups:
        return []
    selected: list[OscarGraphReasoningTask] = []
    offsets = [0 for _ in ordered_groups]
    while len(selected) < max_examples:
        progressed = False
        for group_index, group in enumerate(ordered_groups):
            offset = offsets[group_index]
            if offset >= len(group):
                continue
            selected.append(group[offset])
            offsets[group_index] += 1
            progressed = True
            if len(selected) >= max_examples:
                break
        if not progressed:
            break
    return selected


def build_oscar_graph_reasoning_tasks(
    *,
    max_examples: int | None = None,
    families: Sequence[str] = OSCAR_GRAPH_REASONING_FAMILIES,
) -> tuple[OscarGraphReasoningTask, ...]:
    unsupported = sorted(family for family in families if family not in OSCAR_GRAPH_REASONING_FAMILIES)
    if unsupported:
        raise ValueError(f"Unsupported Oscar graph reasoning families: {unsupported}")
    nodes, edges = build_oscar_canonical_graph()
    task_groups: list[list[OscarGraphReasoningTask]] = []
    if "oscar_graph_relation" in families:
        task_groups.append(_relation_tasks(nodes, edges))
    if "oscar_graph_neighbors" in families:
        task_groups.append(_neighbor_tasks(nodes, edges))
    if "oscar_graph_path_completion" in families:
        task_groups.append(_path_completion_tasks(nodes, edges))
    if "oscar_graph_grounding" in families:
        task_groups.append(_grounding_tasks(nodes, edges))
    if "oscar_graph_executor_rollout" in families:
        task_groups.append(_executor_rollout_tasks(nodes, edges))
    if "oscar_graph_executor_trace" in families:
        task_groups.append(_executor_trace_tasks(nodes, edges))
    return tuple(_round_robin_limit(task_groups, max_examples=max_examples))
