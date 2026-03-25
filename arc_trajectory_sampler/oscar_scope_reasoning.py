from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from .oscar_scope_corpus import OscarScopeRecord, build_oscar_scope_records, resolve_oscar_scope_files
from .state_adapter import FORMAT_HEADER


OSCAR_SCOPE_REASONING_BENCHMARK = "oscar_scope_reasoning"
OSCAR_SCOPE_REASONING_FAMILIES = (
    "oscar_section_anchor",
    "oscar_outline_next_heading",
    "oscar_concept_tags",
    "oscar_workflow_environment",
    "oscar_workflow_kpi_tags",
    "oscar_workflow_bottleneck_tags",
    "oscar_workflow_improvement_tags",
    "oscar_workflow_kpi_improvement",
    "oscar_workflow_intervention_trace",
    "oscar_workflow_case_analogy",
    "oscar_workflow_transfer",
)
OSCAR_WORKFLOW_REASONING_FAMILIES = (
    "oscar_workflow_environment",
    "oscar_workflow_kpi_tags",
    "oscar_workflow_bottleneck_tags",
    "oscar_workflow_improvement_tags",
    "oscar_workflow_kpi_improvement",
    "oscar_workflow_intervention_trace",
    "oscar_workflow_case_analogy",
    "oscar_workflow_transfer",
)

_CONCEPT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("process_intelligence_graph", ("typed temporal property graph", "process intelligence graph", "single primitive")),
    ("oscar_agent", (" oscar", "oscar_", "oscar/kurt agents")),
    ("kurt_agent", (" kurt", "kurt_", "oscar/kurt agents")),
    ("global_practice_model", ("global foundation model", "practice compiler", "f_θ", "f_theta", "global learning")),
    ("intervention_templates", ("intervention templates", "intervention template")),
    ("monitoring_templates", ("monitoring templates", "monitoring template", "monitor bundle")),
    ("permission_learning", ("permission-aware", "privacy budget", "federated learning", "sanitized export")),
    ("thesis_compiler", ("thesis compiler",)),
    ("clean_room_scan", ("clean-room scan", "clean room scan")),
    ("shared_diligence_graph", ("shared diligence graph", "diligence graph")),
    ("day1_baseline", ("day-1 baseline", "day 1 baseline")),
    ("portfolio_value_os", ("portfolio value os",)),
    ("pe_internal_os", ("pe internal os",)),
    ("practice_learning_layer", ("practice learning layer",)),
    ("live_codebase", ("live codebase", "source control", "ci/cd", "deployment", "runtime", "incident response")),
    ("bugfix_lifecycle", ("bug testing", "fix verification", "repair promotion", "repair proposal", "bug-domain")),
    ("recursive_hierarchy", ("recursive abstraction", "hierarchical bugfix", "graph-of-graphs", "layered graphs", "typed temporal contexts")),
    ("visibility_trust_boundary", ("trust boundary", "visibility relation", "data zones", "artifact release")),
)
OSCAR_SCOPE_CONCEPT_TAGS = tuple(tag for tag, _keywords in _CONCEPT_RULES)


@dataclass(frozen=True)
class OscarScopeReasoningTask:
    task_id: str
    benchmark: str
    family: str
    trace_step: str
    text: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class OscarRelatedDocumentLink:
    doc_id: str
    doc_title: str
    reasons: tuple[str, ...]
    shared_concepts: tuple[str, ...]


@dataclass(frozen=True)
class OscarDocumentGraphEntry:
    doc_id: str
    doc_title: str
    doc_group: str
    source_kind: str
    section_paths: tuple[str, ...]
    concept_tags: tuple[str, ...]
    related_documents: tuple[OscarRelatedDocumentLink, ...]


@dataclass(frozen=True)
class WorkflowTaxonomyEntry:
    tag_id: str
    label: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowProfile:
    profile_id: str
    doc_id_fragments: tuple[str, ...]
    environment_label: str
    business_domain: str
    workflow_label: str
    default_kpis: tuple[str, ...]
    default_bottlenecks: tuple[str, ...]
    default_improvements: tuple[str, ...]
    primary_pairs: tuple[tuple[str, str], ...]
    stage_rules: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class WorkflowInterventionPlan:
    focus_kpi_id: str
    primary_improvement_id: str
    followup_improvement_id: str
    active_bottleneck_ids: tuple[str, ...]
    kpi_candidate_ids: tuple[str, ...]
    improvement_candidate_ids: tuple[str, ...]
    primary_reward_score: float
    final_reward_score: float
    primary_reward_bucket_id: int
    final_reward_bucket_id: int
    primary_reward_bucket_label: str
    final_reward_bucket_label: str
    primary_reward_components: dict[str, float]
    final_reward_components: dict[str, float]
    motif_id: str
    motif_label: str


@dataclass(frozen=True)
class WorkflowCaseFrame:
    frame_id: str
    record: OscarScopeRecord
    profile: WorkflowProfile
    doc_title: str
    section_path: str
    body: str
    source_text: str
    stage_label: str
    concept_tags: tuple[str, ...]
    kpi_tags: tuple[str, ...]
    bottleneck_tags: tuple[str, ...]
    improvement_tags: tuple[str, ...]
    plan: WorkflowInterventionPlan


_WORKFLOW_DOCUMENT_STEMS = (
    "genai_workflow_case_study",
    "pepsico_example",
    "vc_portfolio_case_study",
    "pe_workflow_design_integrated_coherent",
    "pe_workflow_design_integrated",
    "pe_workflow_design",
)

_WORKFLOW_KPI_RULES: tuple[WorkflowTaxonomyEntry, ...] = (
    WorkflowTaxonomyEntry("time_to_resolution", "Time to resolution", ("resolve", "ticket", "sendresponse", "customer support", "response")),
    WorkflowTaxonomyEntry("safe_response_rate", "Safe response rate", ("safety", "guardrail", "policy", "brand", "pii")),
    WorkflowTaxonomyEntry("approval_latency", "Approval latency", ("approval", "approve", "committee", "ic meeting", "gate")),
    WorkflowTaxonomyEntry("refund_cycle_time", "Refund cycle time", ("refund",)),
    WorkflowTaxonomyEntry("time_to_invoice", "Time to invoice", ("invoice", "invoice within 24h", "sendinvoice", "createinvoice")),
    WorkflowTaxonomyEntry("manual_touch_rate", "Manual touch rate", ("manual drafting", "manual price correction", "manual-touch", "manual touch")),
    WorkflowTaxonomyEntry("invoice_accuracy", "Invoice accuracy", ("contract price", "price correction", "invoice")),
    WorkflowTaxonomyEntry("days_sales_outstanding", "Days sales outstanding", ("payment", "dso", "receivepayment", "clearpayment")),
    WorkflowTaxonomyEntry("time_to_fill", "Time to fill", ("time-to-fill", "hiring", "offeraccepted", "openrole")),
    WorkflowTaxonomyEntry("time_to_decision", "Time to decision", ("time-to-decision", "follow-on", "allocation decision", "memo", "ic meeting")),
    WorkflowTaxonomyEntry("partner_effort_load", "Partner effort load", ("partner-time", "effort", "memo", "interview loop")),
    WorkflowTaxonomyEntry("diligence_cycle_time", "Diligence cycle time", ("diligence", "request-room", "workstream", "scan")),
    WorkflowTaxonomyEntry("day1_readiness", "Day-1 readiness", ("day-1", "day 1", "baseline", "readiness")),
    WorkflowTaxonomyEntry("value_realization_rate", "Value realization rate", ("value realization", "realized value", "improvement program", "expansion")),
    WorkflowTaxonomyEntry("coverage_completeness", "Coverage completeness", ("coverage map", "ownership map", "outside-in map", "coverage")),
    WorkflowTaxonomyEntry("monitoring_adoption_rate", "Monitoring adoption rate", ("monitor", "adherence", "adoption")),
)

_WORKFLOW_BOTTLENECK_RULES: tuple[WorkflowTaxonomyEntry, ...] = (
    WorkflowTaxonomyEntry("fragmented_handoffs", "Fragmented handoffs", ("disconnected documents", "siloed", "manual drafting", "handoff")),
    WorkflowTaxonomyEntry("approval_queue_delay", "Approval queue delay", ("approval", "approve", "committee", "ic meeting", "gate")),
    WorkflowTaxonomyEntry("policy_guardrail_failure", "Policy or guardrail failure", ("guardrail", "policy", "brand", "pii", "safety")),
    WorkflowTaxonomyEntry("retrieval_context_gap", "Retrieval or context gap", ("retrieval", "rag", "knowledge article", "retrieved docs")),
    WorkflowTaxonomyEntry("pricing_exception_rework", "Pricing exception rework", ("price correction", "contract price")),
    WorkflowTaxonomyEntry("credit_check_delay", "Credit-check delay", ("credit check",)),
    WorkflowTaxonomyEntry("invoice_delay_after_delivery", "Invoice delay after delivery", ("invoice within 24h", "proof-of-delivery", "recordpod", "delivery")),
    WorkflowTaxonomyEntry("interview_scheduling_delay", "Interview scheduling delay", ("interviewscheduled", "interview loop", "sourcecandidate")),
    WorkflowTaxonomyEntry("memo_decision_latency", "Memo or decision latency", ("memo", "follow-on", "allocation decision", "ic meeting")),
    WorkflowTaxonomyEntry("visibility_boundary", "Visibility or trust-boundary friction", ("trust boundary", "visibility relation", "data zones", "clean-room", "release operator")),
    WorkflowTaxonomyEntry("baseline_coverage_gap", "Baseline coverage gap", ("coverage map", "baseline", "ownership map", "outside-in map")),
    WorkflowTaxonomyEntry("initiative_sequencing_risk", "Initiative sequencing risk", ("correct sequence", "sequence", "simplification", "spend control", "ai deployment", "expansion")),
    WorkflowTaxonomyEntry("fragmented_diligence", "Fragmented diligence", ("diligence", "workstream", "expert calls", "request-room")),
)

_WORKFLOW_IMPROVEMENT_RULES: tuple[WorkflowTaxonomyEntry, ...] = (
    WorkflowTaxonomyEntry("rag_copilot", "RAG copilot", ("rag", "llm draft", "copilot", "generate draft")),
    WorkflowTaxonomyEntry("guardrails_human_approval", "Guardrails with human approval", ("guardrail", "human-in-the-loop", "1-click approval", "safety check", "agentapprove")),
    WorkflowTaxonomyEntry("offline_eval_prompt_updates", "Offline eval and prompt updates", ("offlineeval", "updateprompt", "monitor safety", "experiment", "ab exp")),
    WorkflowTaxonomyEntry("invoice_within_24h", "Invoice within 24h", ("invoice within 24h", "createinvoice")),
    WorkflowTaxonomyEntry("trusted_customer_credit_bypass", "Trusted-customer credit bypass", ("skip credit check", "trusted customers", "credit check")),
    WorkflowTaxonomyEntry("contract_price_enforcement", "Contract price enforcement", ("contract price enforcement", "contract price", "price correction")),
    WorkflowTaxonomyEntry("talent_os", "Talent OS", ("talent os", "hiring pipeline", "critical roles")),
    WorkflowTaxonomyEntry("follow_on_sla", "Follow-on SLA", ("follow-on sla", "follow-on", "time-to-decision")),
    WorkflowTaxonomyEntry("board_operating_cadence", "Board operating cadence", ("board/operating cadence", "board meeting", "operating cadence")),
    WorkflowTaxonomyEntry("thesis_compiler", "Thesis compiler", ("thesis compiler",)),
    WorkflowTaxonomyEntry("clean_room_scan", "Clean-room scan", ("clean-room scan", "clean room scan")),
    WorkflowTaxonomyEntry("shared_diligence_graph", "Shared diligence graph", ("shared diligence graph", "diligence graph")),
    WorkflowTaxonomyEntry("day1_baseline_builder", "Day-1 baseline builder", ("day-1 baseline", "day 1 baseline", "baseline builder")),
    WorkflowTaxonomyEntry("portfolio_value_os", "Portfolio Value OS", ("portfolio value os", "value operating system", "post-close")),
    WorkflowTaxonomyEntry("pe_internal_os", "PE Internal OS", ("pe internal operating system", "pe internal os")),
    WorkflowTaxonomyEntry("practice_learning_layer", "Practice learning layer", ("practice learning layer", "sanitized effect reports", "global learning")),
    WorkflowTaxonomyEntry("monitoring_templates", "Monitoring templates", ("monitoring template", "monitor bundles", "monitoring")),
)
OSCAR_WORKFLOW_KPI_IDS = tuple(entry.tag_id for entry in _WORKFLOW_KPI_RULES)
OSCAR_WORKFLOW_BOTTLENECK_IDS = tuple(entry.tag_id for entry in _WORKFLOW_BOTTLENECK_RULES)
OSCAR_WORKFLOW_IMPROVEMENT_IDS = tuple(entry.tag_id for entry in _WORKFLOW_IMPROVEMENT_RULES)
OSCAR_WORKFLOW_REWARD_BUCKETS = (
    "weak",
    "developing",
    "solid",
    "strong",
    "transformative",
)
_WORKFLOW_CANONICAL_KPI_RULES: tuple[WorkflowTaxonomyEntry, ...] = (
    WorkflowTaxonomyEntry("throughput_latency", "Throughput / cycle-time", ()),
    WorkflowTaxonomyEntry("decision_latency", "Decision latency", ()),
    WorkflowTaxonomyEntry("quality_compliance", "Quality / compliance", ()),
    WorkflowTaxonomyEntry("manual_effort", "Manual effort", ()),
    WorkflowTaxonomyEntry("cash_conversion", "Cash conversion", ()),
    WorkflowTaxonomyEntry("readiness_coverage", "Readiness / coverage", ()),
    WorkflowTaxonomyEntry("value_realization", "Value realization", ()),
    WorkflowTaxonomyEntry("monitoring_adoption", "Monitoring adoption", ()),
)
OSCAR_WORKFLOW_CANONICAL_KPI_IDS = tuple(entry.tag_id for entry in _WORKFLOW_CANONICAL_KPI_RULES)
_WORKFLOW_KPI_TO_CANONICAL: dict[str, str] = {
    "time_to_resolution": "throughput_latency",
    "refund_cycle_time": "throughput_latency",
    "time_to_invoice": "throughput_latency",
    "time_to_fill": "throughput_latency",
    "diligence_cycle_time": "throughput_latency",
    "approval_latency": "decision_latency",
    "time_to_decision": "decision_latency",
    "safe_response_rate": "quality_compliance",
    "invoice_accuracy": "quality_compliance",
    "manual_touch_rate": "manual_effort",
    "partner_effort_load": "manual_effort",
    "days_sales_outstanding": "cash_conversion",
    "day1_readiness": "readiness_coverage",
    "coverage_completeness": "readiness_coverage",
    "value_realization_rate": "value_realization",
    "monitoring_adoption_rate": "monitoring_adoption",
}
_WORKFLOW_CANONICAL_INTERVENTION_RULES: tuple[WorkflowTaxonomyEntry, ...] = (
    WorkflowTaxonomyEntry("context_enrichment", "Context enrichment", ()),
    WorkflowTaxonomyEntry("policy_enforcement", "Policy enforcement", ()),
    WorkflowTaxonomyEntry("monitoring_feedback", "Monitoring feedback", ()),
    WorkflowTaxonomyEntry("handoff_automation", "Handoff automation", ()),
    WorkflowTaxonomyEntry("queue_acceleration", "Queue acceleration", ()),
    WorkflowTaxonomyEntry("cadence_governance", "Cadence / governance", ()),
    WorkflowTaxonomyEntry("shared_visibility", "Shared visibility layer", ()),
    WorkflowTaxonomyEntry("baseline_mapping", "Baseline mapping", ()),
    WorkflowTaxonomyEntry("operating_system", "Operating-system platform", ()),
)
OSCAR_WORKFLOW_CANONICAL_INTERVENTION_IDS = tuple(
    entry.tag_id for entry in _WORKFLOW_CANONICAL_INTERVENTION_RULES
)
_WORKFLOW_IMPROVEMENT_TO_CANONICAL_INTERVENTION: dict[str, str] = {
    "rag_copilot": "context_enrichment",
    "thesis_compiler": "context_enrichment",
    "guardrails_human_approval": "policy_enforcement",
    "contract_price_enforcement": "policy_enforcement",
    "offline_eval_prompt_updates": "monitoring_feedback",
    "practice_learning_layer": "monitoring_feedback",
    "monitoring_templates": "monitoring_feedback",
    "invoice_within_24h": "handoff_automation",
    "trusted_customer_credit_bypass": "queue_acceleration",
    "follow_on_sla": "queue_acceleration",
    "board_operating_cadence": "cadence_governance",
    "talent_os": "operating_system",
    "clean_room_scan": "shared_visibility",
    "shared_diligence_graph": "shared_visibility",
    "pe_internal_os": "shared_visibility",
    "day1_baseline_builder": "baseline_mapping",
    "portfolio_value_os": "operating_system",
}
_WORKFLOW_MOTIF_RULES: tuple[WorkflowTaxonomyEntry, ...] = (
    WorkflowTaxonomyEntry("context_gap_recovery", "Context-gap recovery", ("retrieval", "context", "scan", "graph", "thesis")),
    WorkflowTaxonomyEntry("queue_latency_reduction", "Queue-latency reduction", ("approval", "queue", "sla", "credit", "decision")),
    WorkflowTaxonomyEntry("policy_enforcement", "Policy enforcement", ("guardrail", "policy", "contract", "pricing", "safety")),
    WorkflowTaxonomyEntry("handoff_compression", "Handoff compression", ("handoff", "invoice", "manual touch", "delivery")),
    WorkflowTaxonomyEntry("shared_visibility_layer", "Shared visibility layer", ("shared graph", "clean room", "visibility", "coverage")),
    WorkflowTaxonomyEntry("sequenced_rollout", "Sequenced rollout", ("day-1", "baseline", "value creation", "sequence")),
    WorkflowTaxonomyEntry("monitoring_feedback_loop", "Monitoring feedback loop", ("monitoring", "feedback", "offline eval", "practice learning")),
    WorkflowTaxonomyEntry("cadence_standardization", "Cadence standardization", ("cadence", "board", "talent os", "operating cadence")),
)
OSCAR_WORKFLOW_MOTIF_IDS = tuple(entry.tag_id for entry in _WORKFLOW_MOTIF_RULES)
OSCAR_WORKFLOW_ACTION_STEP_IDS = (
    "none",
    "select_kpi_family",
    "select_intervention_family",
)

_WORKFLOW_PROFILES: tuple[WorkflowProfile, ...] = (
    WorkflowProfile(
        profile_id="genai_customer_support",
        doc_id_fragments=("genai_workflow_case_study",),
        environment_label="GenAI customer support copilot",
        business_domain="customer_support",
        workflow_label="Customer support copilot with RAG, guardrails, approval, and refunds",
        default_kpis=("time_to_resolution", "safe_response_rate", "approval_latency", "refund_cycle_time"),
        default_bottlenecks=("retrieval_context_gap", "policy_guardrail_failure", "approval_queue_delay"),
        default_improvements=("rag_copilot", "guardrails_human_approval", "offline_eval_prompt_updates"),
        primary_pairs=(
            ("time_to_resolution", "rag_copilot"),
            ("safe_response_rate", "guardrails_human_approval"),
            ("approval_latency", "guardrails_human_approval"),
        ),
        stage_rules=(
            ("retrieval", ("retrieval", "rag", "knowledge article")),
            ("generation", ("generate draft", "generation", "prompt template")),
            ("safety_and_approval", ("safety check", "human-in-the-loop", "approve")),
            ("downstream_action", ("refund", "send response", "resolve ticket")),
            ("offline_eval_and_monitoring", ("offlineeval", "monitor safety", "updateprompt", "experiment")),
        ),
    ),
    WorkflowProfile(
        profile_id="cpg_order_to_cash",
        doc_id_fragments=("pepsico_example",),
        environment_label="CPG order-to-cash",
        business_domain="consumer_goods",
        workflow_label="Consumer-goods order-to-cash from sales order through payment",
        default_kpis=("time_to_invoice", "manual_touch_rate", "invoice_accuracy", "days_sales_outstanding"),
        default_bottlenecks=("credit_check_delay", "pricing_exception_rework", "invoice_delay_after_delivery"),
        default_improvements=("invoice_within_24h", "trusted_customer_credit_bypass", "contract_price_enforcement"),
        primary_pairs=(
            ("time_to_invoice", "invoice_within_24h"),
            ("manual_touch_rate", "contract_price_enforcement"),
            ("days_sales_outstanding", "trusted_customer_credit_bypass"),
        ),
        stage_rules=(
            ("order_capture", ("create sales order", "createsalesorder")),
            ("credit", ("credit check", "creditcheck")),
            ("delivery_and_pod", ("delivery", "goodsissue", "recordpod", "proof-of-delivery")),
            ("invoicing", ("invoice", "createinvoice", "sendinvoice")),
            ("payment_collection", ("payment", "receivepayment", "clearpayment")),
        ),
    ),
    WorkflowProfile(
        profile_id="vc_portfolio_support",
        doc_id_fragments=("vc_portfolio_case_study",),
        environment_label="VC portfolio support",
        business_domain="venture_capital",
        workflow_label="Early-stage VC hiring, follow-on decisions, and board operating cadence",
        default_kpis=("time_to_fill", "time_to_decision", "partner_effort_load", "monitoring_adoption_rate"),
        default_bottlenecks=("interview_scheduling_delay", "memo_decision_latency", "approval_queue_delay"),
        default_improvements=("talent_os", "follow_on_sla", "board_operating_cadence"),
        primary_pairs=(
            ("time_to_fill", "talent_os"),
            ("time_to_decision", "follow_on_sla"),
            ("partner_effort_load", "board_operating_cadence"),
        ),
        stage_rules=(
            ("hiring_pipeline", ("openrole", "candidate", "interview", "offer")),
            ("follow_on_decision", ("follow-on", "memo", "ic meeting", "allocation decision")),
            ("board_operating_cadence", ("board meeting", "operating cadence")),
        ),
    ),
    WorkflowProfile(
        profile_id="private_equity_lifecycle",
        doc_id_fragments=(
            "pe_workflow_design_integrated_coherent",
            "pe_workflow_design_integrated",
            "pe_workflow_design",
        ),
        environment_label="Private-equity lifecycle",
        business_domain="private_equity",
        workflow_label="Thesis-to-value private-equity lifecycle across diligence, Day-1, and value creation",
        default_kpis=("diligence_cycle_time", "day1_readiness", "value_realization_rate", "coverage_completeness"),
        default_bottlenecks=("fragmented_diligence", "visibility_boundary", "baseline_coverage_gap", "initiative_sequencing_risk"),
        default_improvements=("thesis_compiler", "clean_room_scan", "shared_diligence_graph", "day1_baseline_builder", "portfolio_value_os", "practice_learning_layer"),
        primary_pairs=(
            ("diligence_cycle_time", "clean_room_scan"),
            ("coverage_completeness", "shared_diligence_graph"),
            ("day1_readiness", "day1_baseline_builder"),
            ("value_realization_rate", "portfolio_value_os"),
        ),
        stage_rules=(
            ("thesis_compilation", ("thesis compiler", "sector thesis", "deal-specific hypothesis")),
            ("clean_room_scan", ("clean-room scan", "clean room scan")),
            ("shared_diligence", ("shared diligence graph", "diligence", "workstream")),
            ("day1_baseline", ("day-1 baseline", "day 1 baseline", "baseline builder")),
            ("portfolio_value_creation", ("portfolio value os", "value realization", "improvement program")),
            ("pe_internal_ops", ("pe internal os", "internal operating system")),
            ("practice_learning", ("practice learning layer", "sanitized effect reports", "global learning")),
        ),
    ),
)

_WORKFLOW_IMPROVEMENT_STAGE_HINTS: dict[str, tuple[str, ...]] = {
    "rag_copilot": ("retrieval", "generation"),
    "guardrails_human_approval": ("generation", "safety_and_approval"),
    "offline_eval_prompt_updates": ("offline_eval_and_monitoring",),
    "invoice_within_24h": ("delivery_and_pod", "invoicing"),
    "trusted_customer_credit_bypass": ("credit",),
    "contract_price_enforcement": ("order_capture", "invoicing"),
    "talent_os": ("hiring_pipeline",),
    "follow_on_sla": ("follow_on_decision",),
    "board_operating_cadence": ("board_operating_cadence",),
    "thesis_compiler": ("thesis_compilation",),
    "clean_room_scan": ("clean_room_scan",),
    "shared_diligence_graph": ("shared_diligence",),
    "day1_baseline_builder": ("day1_baseline",),
    "portfolio_value_os": ("portfolio_value_creation",),
    "pe_internal_os": ("pe_internal_ops",),
    "practice_learning_layer": ("practice_learning",),
    "monitoring_templates": ("practice_learning", "offline_eval_and_monitoring"),
}

_WORKFLOW_IMPROVEMENT_BOTTLENECK_HINTS: dict[str, tuple[str, ...]] = {
    "rag_copilot": ("retrieval_context_gap", "fragmented_handoffs"),
    "guardrails_human_approval": ("policy_guardrail_failure", "approval_queue_delay"),
    "offline_eval_prompt_updates": ("policy_guardrail_failure", "retrieval_context_gap"),
    "invoice_within_24h": ("invoice_delay_after_delivery",),
    "trusted_customer_credit_bypass": ("credit_check_delay",),
    "contract_price_enforcement": ("pricing_exception_rework",),
    "talent_os": ("interview_scheduling_delay",),
    "follow_on_sla": ("memo_decision_latency", "approval_queue_delay"),
    "board_operating_cadence": ("memo_decision_latency",),
    "thesis_compiler": ("fragmented_diligence", "baseline_coverage_gap"),
    "clean_room_scan": ("fragmented_diligence", "visibility_boundary"),
    "shared_diligence_graph": ("fragmented_diligence", "baseline_coverage_gap"),
    "day1_baseline_builder": ("baseline_coverage_gap", "initiative_sequencing_risk"),
    "portfolio_value_os": ("initiative_sequencing_risk",),
    "pe_internal_os": ("visibility_boundary",),
    "practice_learning_layer": ("visibility_boundary", "initiative_sequencing_risk"),
    "monitoring_templates": ("policy_guardrail_failure", "initiative_sequencing_risk"),
}

_WORKFLOW_IMPROVEMENT_FOLLOWUPS: dict[str, tuple[str, ...]] = {
    "rag_copilot": ("guardrails_human_approval", "offline_eval_prompt_updates"),
    "guardrails_human_approval": ("offline_eval_prompt_updates", "monitoring_templates"),
    "offline_eval_prompt_updates": ("monitoring_templates",),
    "invoice_within_24h": ("contract_price_enforcement",),
    "trusted_customer_credit_bypass": ("invoice_within_24h", "contract_price_enforcement"),
    "contract_price_enforcement": ("invoice_within_24h",),
    "talent_os": ("board_operating_cadence",),
    "follow_on_sla": ("board_operating_cadence",),
    "board_operating_cadence": ("follow_on_sla",),
    "thesis_compiler": ("clean_room_scan", "shared_diligence_graph"),
    "clean_room_scan": ("shared_diligence_graph", "day1_baseline_builder"),
    "shared_diligence_graph": ("day1_baseline_builder", "practice_learning_layer"),
    "day1_baseline_builder": ("portfolio_value_os",),
    "portfolio_value_os": ("practice_learning_layer",),
    "pe_internal_os": ("practice_learning_layer",),
    "practice_learning_layer": ("monitoring_templates",),
    "monitoring_templates": ("practice_learning_layer",),
}
_WORKFLOW_IMPROVEMENT_TO_MOTIF: dict[str, str] = {
    "rag_copilot": "context_gap_recovery",
    "guardrails_human_approval": "policy_enforcement",
    "offline_eval_prompt_updates": "monitoring_feedback_loop",
    "invoice_within_24h": "handoff_compression",
    "trusted_customer_credit_bypass": "queue_latency_reduction",
    "contract_price_enforcement": "policy_enforcement",
    "talent_os": "cadence_standardization",
    "follow_on_sla": "queue_latency_reduction",
    "board_operating_cadence": "cadence_standardization",
    "thesis_compiler": "context_gap_recovery",
    "clean_room_scan": "shared_visibility_layer",
    "shared_diligence_graph": "shared_visibility_layer",
    "day1_baseline_builder": "sequenced_rollout",
    "portfolio_value_os": "sequenced_rollout",
    "pe_internal_os": "shared_visibility_layer",
    "practice_learning_layer": "monitoring_feedback_loop",
    "monitoring_templates": "monitoring_feedback_loop",
}
_WORKFLOW_BOTTLENECK_TO_MOTIF: dict[str, str] = {
    "retrieval_context_gap": "context_gap_recovery",
    "approval_queue_delay": "queue_latency_reduction",
    "policy_guardrail_failure": "policy_enforcement",
    "fragmented_handoffs": "handoff_compression",
    "pricing_exception_rework": "policy_enforcement",
    "credit_check_delay": "queue_latency_reduction",
    "invoice_delay_after_delivery": "handoff_compression",
    "interview_scheduling_delay": "queue_latency_reduction",
    "memo_decision_latency": "queue_latency_reduction",
    "visibility_boundary": "shared_visibility_layer",
    "baseline_coverage_gap": "shared_visibility_layer",
    "initiative_sequencing_risk": "sequenced_rollout",
    "fragmented_diligence": "shared_visibility_layer",
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


def _section_path_leaf(section_path: str) -> str:
    if " > " not in section_path:
        return section_path.strip()
    return section_path.split(" > ")[-1].strip()


def _section_depth(section_path: str) -> int:
    return max(1, len([part for part in section_path.split(" > ") if part.strip()]))


def _section_path_prefixes(section_path: str) -> tuple[str, ...]:
    parts = [part.strip() for part in section_path.split(" > ") if part.strip()]
    return tuple(" > ".join(parts[: index + 1]) for index in range(len(parts)))


def _section_parent_label(section_path: str, *, doc_title: str) -> str:
    prefixes = _section_path_prefixes(section_path)
    if len(prefixes) <= 1:
        return doc_title
    return prefixes[-2]


def _section_ancestor_labels(section_path: str) -> tuple[str, ...]:
    prefixes = _section_path_prefixes(section_path)
    if len(prefixes) <= 1:
        return ()
    return prefixes[:-1]


def _normalized_text_tokens(text: str) -> tuple[str, ...]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    tokens = [token for token in normalized.split() if len(token) > 2]
    return tuple(tokens)


def _shared_token_count(left: str, right: str) -> int:
    return len(set(_normalized_text_tokens(left)).intersection(_normalized_text_tokens(right)))


def _shared_section_prefix_depth(left: str, right: str) -> int:
    left_parts = [part.strip() for part in left.split(" > ") if part.strip()]
    right_parts = [part.strip() for part in right.split(" > ") if part.strip()]
    depth = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        depth += 1
    return depth


def _parse_outline_entries(record_text: str) -> tuple[tuple[int, str], ...]:
    entries: list[tuple[int, str]] = []
    _header, body = _split_scope_text(record_text)
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            continue
        indent = len(line) - len(stripped)
        level = 1 + (indent // 2)
        title = stripped[2:].strip()
        if title:
            entries.append((level, title))
    return tuple(entries)


def _json_line(key: str, value: object) -> str:
    return f"{key}=" + json.dumps(value, separators=(",", ":"), sort_keys=True)


def _outline_target_section_path(
    doc_title: str,
    prefix: Sequence[tuple[int, str]],
    *,
    next_level: int,
    next_title: str,
) -> str:
    stack: list[str] = [doc_title]
    for level, title in (*prefix, (next_level, next_title)):
        stack = stack[: max(1, level)]
        stack.append(title)
    return " > ".join(part for part in stack if part)


def _state_tokens(
    *,
    doc_group: str,
    source_kind: str,
    source_view: str,
    family: str,
    section_depth: int,
    task_kind: str,
) -> tuple[str, ...]:
    return (
        f"doc.group={doc_group}",
        f"source.kind={source_kind}",
        f"source.view={source_view}",
        f"family={family}",
        f"task.kind={task_kind}",
        f"section.depth={section_depth}",
    )


def _serialize_reasoning_task(
    *,
    task_id: str,
    family: str,
    difficulty: int,
    trace_step: str,
    answer_format: str,
    doc_title: str,
    doc_group: str,
    doc_id: str,
    source_kind: str,
    source_view: str,
    section_path: str,
    question: str,
    context_payload: dict[str, object],
    target_answer: object,
    auxiliary_targets: dict[str, object],
    task_kind: str,
) -> str:
    section_depth = _section_depth(section_path)
    state_tokens = _state_tokens(
        doc_group=doc_group,
        source_kind=source_kind,
        source_view=source_view,
        family=family,
        section_depth=section_depth,
        task_kind=task_kind,
    )
    state_scalars = (
        float(section_depth),
        float(len(context_payload.get("excerpt", "") if isinstance(context_payload.get("excerpt"), str) else "")),
        float(len(context_payload.get("outline_prefix", ()) if isinstance(context_payload.get("outline_prefix"), (list, tuple)) else ())),
    )
    lines = [
        FORMAT_HEADER,
        "record_type=terminal_answer",
        f"dataset={OSCAR_SCOPE_REASONING_BENCHMARK}",
        "source_modality=document_scope",
        f"family={family}",
        f"difficulty={difficulty}",
        f"trajectory_id={task_id}",
        "step_index=0",
        "trace_length=1",
        f"trace_step={trace_step}",
        "previous_action=<start>",
        f"candidate_bucket=family:{family}|step:0|prev:<start>",
        f"answer_format={answer_format}",
        "variant_kind=domain_reasoning",
        "trajectory_role=gold",
        f"doc_group={doc_group}",
        f"doc_title={doc_title}",
        f"doc_id={doc_id}",
        f"source_kind={source_kind}",
        f"source_view={source_view}",
        f"section_path={section_path}",
        "state_tokens=" + " ".join(state_tokens),
        "state_scalars=" + ",".join(f"{value:.6f}" for value in state_scalars),
        _json_line("question", question),
        _json_line("context", context_payload),
        _json_line("auxiliary_targets", auxiliary_targets),
        "target_answer=" + json.dumps(target_answer, separators=(",", ":"), sort_keys=True),
    ]
    return "\n".join(lines) + "\n"


def _serialize_workflow_decision_step(
    *,
    task_id: str,
    family: str,
    difficulty: int,
    trace_step: str,
    step_index: int,
    trace_length: int,
    previous_action: str,
    candidate_bucket: str,
    doc_title: str,
    doc_group: str,
    doc_id: str,
    source_kind: str,
    source_view: str,
    section_path: str,
    question: str,
    context_payload: dict[str, object],
    verifier_context: dict[str, object],
    auxiliary_targets: dict[str, object],
    target_action: dict[str, object],
    task_kind: str,
    state_tokens: Sequence[str],
    state_scalars: Sequence[float],
) -> str:
    lines = [
        FORMAT_HEADER,
        "record_type=decision_action",
        f"dataset={OSCAR_SCOPE_REASONING_BENCHMARK}",
        "source_modality=document_scope",
        f"family={family}",
        f"difficulty={difficulty}",
        f"trajectory_id={task_id}",
        f"step_index={step_index}",
        f"trace_length={trace_length}",
        f"trace_step={trace_step}",
        f"previous_action={previous_action}",
        f"candidate_bucket={candidate_bucket}",
        "answer_format=json",
        "variant_kind=domain_workflow_executor",
        "trajectory_role=gold",
        f"doc_group={doc_group}",
        f"doc_title={doc_title}",
        f"doc_id={doc_id}",
        f"source_kind={source_kind}",
        f"source_view={source_view}",
        f"section_path={section_path}",
        "state_tokens=" + " ".join(state_tokens),
        "state_scalars=" + ",".join(f"{value:.6f}" for value in state_scalars),
        _json_line("verifier_context", verifier_context),
        _json_line("question", question),
        _json_line("context", context_payload),
        _json_line("auxiliary_targets", auxiliary_targets),
        _json_line("target_action", target_action),
    ]
    return "\n".join(lines) + "\n"


def _native_section_anchor_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> list[OscarScopeReasoningTask]:
    tasks: list[OscarScopeReasoningTask] = []
    for record in records:
        if record.view != "native_chunk":
            continue
        header, body = _split_scope_text(record.text)
        section_path = str(record.metadata.get("section_path", header.get("section_path", ""))).strip()
        if not section_path or not body:
            continue
        leaf = _section_path_leaf(section_path)
        doc_title = str(record.metadata.get("doc_title", header.get("doc_title", record.document_id)))
        concept_tags = _concept_tags_for_text(" ".join(part for part in (doc_title, section_path, body) if part))
        if leaf == doc_title:
            continue
        task_id = f"{record.document_id}:section_anchor:{record.chunk_index}"
        auxiliary_targets = _augment_task_metadata(
            {
            "family": "oscar_section_anchor",
            "task_kind": "section_anchor",
            "doc_group": str(record.metadata.get("doc_group", "unknown")),
            "doc_title": doc_title,
            "doc_id": record.document_id,
            "source_view": record.view,
            "section_path": section_path,
            "section_depth": _section_depth(section_path),
            "section_path_leaf": leaf,
            "section_path_label": section_path,
            "section_anchor": leaf,
            "concept_tags": list(concept_tags),
            },
            graph_index=graph_index,
        )
        text = _serialize_reasoning_task(
            task_id=task_id,
            family="oscar_section_anchor",
            difficulty=min(5, _section_depth(section_path)),
            trace_step="reason",
            answer_format="string",
            doc_title=doc_title,
            doc_group=str(record.metadata.get("doc_group", "unknown")),
            doc_id=record.document_id,
            source_kind=str(record.metadata.get("source_kind", "unknown")),
            source_view=record.view,
            section_path=section_path,
            question="Which section heading best matches this Oscar-scope excerpt?",
            context_payload={
                "excerpt": body,
                "doc_title": doc_title,
            },
            target_answer=leaf,
            auxiliary_targets=auxiliary_targets,
            task_kind="section_anchor",
        )
        tasks.append(
            OscarScopeReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family="oscar_section_anchor",
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return tasks


def _outline_next_heading_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> list[OscarScopeReasoningTask]:
    tasks: list[OscarScopeReasoningTask] = []
    for record in records:
        if record.view != "section_outline":
            continue
        entries = _parse_outline_entries(record.text)
        if len(entries) < 2:
            continue
        doc_title = str(record.metadata.get("doc_title", record.document_id))
        for index in range(1, len(entries)):
            prefix = entries[:index]
            next_level, next_title = entries[index]
            section_path = _outline_target_section_path(
                doc_title,
                prefix,
                next_level=next_level,
                next_title=next_title,
            )
            concept_tags = _concept_tags_for_text(
                " ".join(
                    [doc_title]
                    + [title for _level, title in prefix]
                    + [next_title]
                )
            )
            prefix_payload = [
                {
                    "level": level,
                    "title": title,
                }
                for level, title in prefix
            ]
            task_id = f"{record.document_id}:outline_next:{index}"
            auxiliary_targets = _augment_task_metadata(
                {
                "family": "oscar_outline_next_heading",
                "task_kind": "outline_next_heading",
                "doc_group": str(record.metadata.get("doc_group", "unknown")),
                "doc_title": doc_title,
                "doc_id": record.document_id,
                "source_view": record.view,
                "outline_prefix_length": len(prefix),
                "next_heading_level": next_level,
                "section_path": section_path,
                "section_depth": _section_depth(section_path),
                "section_path_leaf": next_title,
                "section_path_label": section_path,
                "concept_tags": list(concept_tags),
                },
                graph_index=graph_index,
            )
            text = _serialize_reasoning_task(
                task_id=task_id,
                family="oscar_outline_next_heading",
                difficulty=min(5, next_level + max(0, len(prefix) // 4)),
                trace_step="reason",
                answer_format="json",
                doc_title=doc_title,
                doc_group=str(record.metadata.get("doc_group", "unknown")),
                doc_id=record.document_id,
                source_kind=str(record.metadata.get("source_kind", "unknown")),
                source_view=record.view,
                section_path=section_path,
                question="What heading comes next in this Oscar-scope outline?",
                context_payload={
                    "outline_prefix": prefix_payload,
                    "doc_title": doc_title,
                },
                target_answer={"level": next_level, "title": next_title},
                auxiliary_targets=auxiliary_targets,
                task_kind="outline_next_heading",
            )
            tasks.append(
                OscarScopeReasoningTask(
                    task_id=task_id,
                    benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                    family="oscar_outline_next_heading",
                    trace_step="reason",
                    text=text,
                    metadata=auxiliary_targets,
                )
            )
    return tasks


def _concept_tags_for_text(text: str) -> tuple[str, ...]:
    lowered = f" {text.lower()} "
    tags = [
        tag
        for tag, keywords in _CONCEPT_RULES
        if any(keyword in lowered for keyword in keywords)
    ]
    return tuple(sorted(tags))


def _outline_section_paths(record: OscarScopeRecord) -> tuple[str, ...]:
    entries = _parse_outline_entries(record.text)
    if len(entries) < 1:
        return ()
    doc_title = str(record.metadata.get("doc_title", record.document_id))
    paths = []
    for index, (next_level, next_title) in enumerate(entries):
        prefix = entries[:index]
        paths.append(
            _outline_target_section_path(
                doc_title,
                prefix,
                next_level=next_level,
                next_title=next_title,
            )
        )
    return tuple(paths)


def _build_document_graph_index(records: Sequence[OscarScopeRecord]) -> dict[str, OscarDocumentGraphEntry]:
    doc_titles: dict[str, str] = {}
    doc_groups: dict[str, str] = {}
    source_kinds: dict[str, str] = {}
    section_paths_by_doc: dict[str, set[str]] = {}
    concept_tags_by_doc: dict[str, set[str]] = {}

    for record in records:
        doc_id = record.document_id
        doc_title = str(record.metadata.get("doc_title", record.document_id))
        doc_titles[doc_id] = doc_title
        doc_groups[doc_id] = str(record.metadata.get("doc_group", "unknown"))
        source_kinds[doc_id] = str(record.metadata.get("source_kind", "unknown"))
        section_paths_by_doc.setdefault(doc_id, set()).add(doc_title)
        concept_tags_by_doc.setdefault(doc_id, set())

        if record.view == "native_chunk":
            header, body = _split_scope_text(record.text)
            section_path = str(record.metadata.get("section_path", header.get("section_path", doc_title))).strip() or doc_title
            section_paths_by_doc[doc_id].add(section_path)
            concept_tags_by_doc[doc_id].update(
                _concept_tags_for_text(" ".join(part for part in (doc_title, section_path, body) if part))
            )
        elif record.view == "section_outline":
            for section_path in _outline_section_paths(record):
                section_paths_by_doc[doc_id].add(section_path)
            concept_tags_by_doc[doc_id].update(
                _concept_tags_for_text(" ".join(part for part in (doc_title, record.text) if part))
            )

    sorted_doc_ids = tuple(sorted(doc_titles))
    related_documents_by_doc: dict[str, tuple[OscarRelatedDocumentLink, ...]] = {}
    for doc_id in sorted_doc_ids:
        current_group = doc_groups.get(doc_id, "unknown")
        current_concepts = concept_tags_by_doc.get(doc_id, set())
        links: list[OscarRelatedDocumentLink] = []
        for other_doc_id in sorted_doc_ids:
            if other_doc_id == doc_id:
                continue
            reasons: list[str] = []
            if doc_groups.get(other_doc_id, "unknown") == current_group:
                reasons.append("same_group")
            shared_concepts = tuple(
                sorted(current_concepts.intersection(concept_tags_by_doc.get(other_doc_id, set())))
            )
            if shared_concepts:
                reasons.append("shared_concept")
            if not reasons:
                continue
            links.append(
                OscarRelatedDocumentLink(
                    doc_id=other_doc_id,
                    doc_title=doc_titles.get(other_doc_id, other_doc_id),
                    reasons=tuple(reasons),
                    shared_concepts=shared_concepts,
                )
            )
        links.sort(
            key=lambda item: (
                -len(item.shared_concepts),
                "same_group" not in item.reasons,
                item.doc_title.lower(),
                item.doc_id,
            )
        )
        related_documents_by_doc[doc_id] = tuple(links)

    return {
        doc_id: OscarDocumentGraphEntry(
            doc_id=doc_id,
            doc_title=doc_titles.get(doc_id, doc_id),
            doc_group=doc_groups.get(doc_id, "unknown"),
            source_kind=source_kinds.get(doc_id, "unknown"),
            section_paths=tuple(sorted(section_paths_by_doc.get(doc_id, {doc_titles.get(doc_id, doc_id)}))),
            concept_tags=tuple(sorted(concept_tags_by_doc.get(doc_id, set()))),
            related_documents=related_documents_by_doc.get(doc_id, ()),
        )
        for doc_id in sorted_doc_ids
    }


def _local_graph_payload(
    *,
    doc_entry: OscarDocumentGraphEntry,
    section_path: str,
    concept_tags: Sequence[str],
) -> dict[str, object]:
    section_prefixes = _section_path_prefixes(section_path) or (doc_entry.doc_title,)
    local_concepts = tuple(sorted({str(tag) for tag in concept_tags if str(tag)}))
    local_nodes: list[dict[str, object]] = [
        {
            "id": f"doc:{doc_entry.doc_id}",
            "type": "document",
            "doc_id": doc_entry.doc_id,
            "doc_title": doc_entry.doc_title,
            "doc_group": doc_entry.doc_group,
        }
    ]
    local_edges: list[dict[str, object]] = []

    for prefix in section_prefixes:
        local_nodes.append(
            {
                "id": f"section:{doc_entry.doc_id}:{prefix}",
                "type": "section",
                "doc_id": doc_entry.doc_id,
                "section_path": prefix,
                "depth": _section_depth(prefix),
            }
        )
        local_edges.append(
            {
                "source": f"doc:{doc_entry.doc_id}",
                "relation": "contains_section",
                "target": f"section:{doc_entry.doc_id}:{prefix}",
            }
        )
    for parent, child in zip(section_prefixes[:-1], section_prefixes[1:]):
        local_edges.append(
            {
                "source": f"section:{doc_entry.doc_id}:{parent}",
                "relation": "parent_section",
                "target": f"section:{doc_entry.doc_id}:{child}",
            }
        )
    current_section_node = f"section:{doc_entry.doc_id}:{section_prefixes[-1]}"
    for concept_tag in local_concepts:
        local_nodes.append(
            {
                "id": f"concept:{concept_tag}",
                "type": "concept",
                "concept_tag": concept_tag,
            }
        )
        local_edges.append(
            {
                "source": current_section_node,
                "relation": "mentions_concept",
                "target": f"concept:{concept_tag}",
            }
        )

    for related_doc in doc_entry.related_documents[:8]:
        related_node_id = f"doc:{related_doc.doc_id}"
        local_nodes.append(
            {
                "id": related_node_id,
                "type": "document",
                "doc_id": related_doc.doc_id,
                "doc_title": related_doc.doc_title,
            }
        )
        local_edges.append(
            {
                "source": f"doc:{doc_entry.doc_id}",
                "relation": "related_document",
                "target": related_node_id,
                "reasons": list(related_doc.reasons),
                "shared_concepts": list(related_doc.shared_concepts),
            }
        )
    return {
        "nodes": local_nodes,
        "edges": local_edges,
    }


def _augment_task_metadata(
    auxiliary_targets: dict[str, object],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> dict[str, object]:
    doc_id = str(auxiliary_targets.get("doc_id", ""))
    doc_entry = graph_index.get(doc_id)
    if doc_entry is None:
        return auxiliary_targets
    doc_title = str(auxiliary_targets.get("doc_title", doc_entry.doc_title))
    section_path = str(auxiliary_targets.get("section_path_label", auxiliary_targets.get("section_path", doc_title))).strip() or doc_title
    concept_tags = tuple(
        sorted(
            {
                str(tag)
                for tag in auxiliary_targets.get("concept_tags", ())
                if str(tag) in OSCAR_SCOPE_CONCEPT_TAGS
            }
        )
    )
    section_parent_label = _section_parent_label(section_path, doc_title=doc_title)
    section_ancestor_labels = _section_ancestor_labels(section_path)
    related_documents = doc_entry.related_documents
    return {
        **auxiliary_targets,
        "section_parent_label": section_parent_label,
        "section_ancestor_labels": list(section_ancestor_labels),
        "related_doc_titles": [link.doc_title for link in related_documents],
        "related_doc_ids": [link.doc_id for link in related_documents],
        "document_link_reasons": {
            link.doc_title: {
                "reasons": list(link.reasons),
                "shared_concepts": list(link.shared_concepts),
            }
            for link in related_documents
        },
        "local_graph": _local_graph_payload(
            doc_entry=doc_entry,
            section_path=section_path,
            concept_tags=concept_tags,
        ),
    }


def _concept_tag_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> list[OscarScopeReasoningTask]:
    tasks: list[OscarScopeReasoningTask] = []
    for record in records:
        if record.view != "native_chunk":
            continue
        header, body = _split_scope_text(record.text)
        section_path = str(record.metadata.get("section_path", header.get("section_path", ""))).strip()
        doc_title = str(record.metadata.get("doc_title", header.get("doc_title", record.document_id)))
        source_text = " ".join(
            part
            for part in (
                doc_title,
                section_path,
                body,
            )
            if part
        )
        tags = _concept_tags_for_text(source_text)
        if not tags:
            continue
        task_id = f"{record.document_id}:concept_tags:{record.chunk_index}"
        auxiliary_targets = _augment_task_metadata(
            {
            "family": "oscar_concept_tags",
            "task_kind": "concept_tags",
            "doc_group": str(record.metadata.get("doc_group", "unknown")),
            "doc_title": doc_title,
            "doc_id": record.document_id,
            "source_view": record.view,
            "section_path": section_path,
            "section_depth": _section_depth(section_path or doc_title),
            "section_path_leaf": _section_path_leaf(section_path or doc_title),
            "section_path_label": section_path or doc_title,
            "concept_tags": list(tags),
            },
            graph_index=graph_index,
        )
        text = _serialize_reasoning_task(
            task_id=task_id,
            family="oscar_concept_tags",
            difficulty=min(5, max(1, len(tags))),
            trace_step="reason",
            answer_format="json",
            doc_title=doc_title,
            doc_group=str(record.metadata.get("doc_group", "unknown")),
            doc_id=record.document_id,
            source_kind=str(record.metadata.get("source_kind", "unknown")),
            source_view=record.view,
            section_path=section_path or doc_title,
            question="Which canonical Oscar concepts are central in this excerpt?",
            context_payload={
                "excerpt": body,
                "doc_title": doc_title,
            },
            target_answer=list(tags),
            auxiliary_targets=auxiliary_targets,
            task_kind="concept_tags",
        )
        tasks.append(
            OscarScopeReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family="oscar_concept_tags",
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return tasks


def _merge_records(*groups: Sequence[OscarScopeRecord]) -> tuple[OscarScopeRecord, ...]:
    merged: dict[tuple[str, int, str, str], OscarScopeRecord] = {}
    for group in groups:
        for record in group:
            key = (record.document_id, record.chunk_index, record.view, record.text)
            if key not in merged:
                merged[key] = record
    return tuple(merged.values())


def _workflow_scope_records(
    *,
    roots: Sequence[str | Path],
    paths: Sequence[str | Path],
    auto_discover: bool,
    views: Sequence[str],
) -> tuple[OscarScopeRecord, ...]:
    workflow_files = tuple(
        path
        for path in resolve_oscar_scope_files(roots=roots, paths=paths, auto_discover=auto_discover)
        if path.stem in _WORKFLOW_DOCUMENT_STEMS
    )
    if not workflow_files:
        return ()
    return build_oscar_scope_records(
        paths=workflow_files,
        auto_discover=False,
        max_documents=None,
        max_chunks=None,
        views=views,
    )


def _workflow_profile_for_document(*, doc_id: str, doc_title: str) -> WorkflowProfile | None:
    lowered_doc_id = doc_id.lower()
    lowered_title = doc_title.lower()
    for profile in _WORKFLOW_PROFILES:
        if any(fragment in lowered_doc_id or fragment in lowered_title for fragment in profile.doc_id_fragments):
            return profile
    return None


def _workflow_taxonomy_label_map(entries: Sequence[WorkflowTaxonomyEntry]) -> dict[str, str]:
    return {entry.tag_id: entry.label for entry in entries}


def _workflow_motif_label_map() -> dict[str, str]:
    return _workflow_taxonomy_label_map(_WORKFLOW_MOTIF_RULES)


def _workflow_canonical_kpi_label_map() -> dict[str, str]:
    return _workflow_taxonomy_label_map(_WORKFLOW_CANONICAL_KPI_RULES)


def _workflow_canonical_intervention_label_map() -> dict[str, str]:
    return _workflow_taxonomy_label_map(_WORKFLOW_CANONICAL_INTERVENTION_RULES)


def _workflow_canonical_kpi_id(kpi_id: str) -> str:
    return _WORKFLOW_KPI_TO_CANONICAL.get(kpi_id, "throughput_latency")


def _workflow_canonical_intervention_id(improvement_id: str) -> str:
    return _WORKFLOW_IMPROVEMENT_TO_CANONICAL_INTERVENTION.get(improvement_id, "operating_system")


def _workflow_ranked_tags(
    text: str,
    *,
    entries: Sequence[WorkflowTaxonomyEntry],
    defaults: Sequence[str],
    limit: int = 4,
) -> tuple[str, ...]:
    lowered = f" {text.lower()} "
    scored: list[tuple[str, int, int]] = []
    for index, entry in enumerate(entries):
        score = sum(1 for keyword in entry.keywords if keyword in lowered)
        if score > 0:
            scored.append((entry.tag_id, score, index))
    scored.sort(key=lambda item: (-item[1], item[2], item[0]))
    selected: list[str] = []
    for tag_id, _score, _index in scored:
        if tag_id not in selected:
            selected.append(tag_id)
        if len(selected) >= limit:
            return tuple(selected)
    for tag_id in defaults:
        if tag_id not in selected:
            selected.append(tag_id)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _workflow_unique_ids(values: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _workflow_reward_bucket(score: float) -> tuple[int, str]:
    if score < 0.25:
        return 0, OSCAR_WORKFLOW_REWARD_BUCKETS[0]
    if score < 0.45:
        return 1, OSCAR_WORKFLOW_REWARD_BUCKETS[1]
    if score < 0.65:
        return 2, OSCAR_WORKFLOW_REWARD_BUCKETS[2]
    if score < 0.82:
        return 3, OSCAR_WORKFLOW_REWARD_BUCKETS[3]
    return 4, OSCAR_WORKFLOW_REWARD_BUCKETS[4]


def _workflow_stage_fit_score(improvement_id: str, stage_label: str) -> float:
    if stage_label == "workflow_overview":
        return 0.6
    hinted_stages = _WORKFLOW_IMPROVEMENT_STAGE_HINTS.get(improvement_id, ())
    if not hinted_stages:
        return 0.5
    if stage_label in hinted_stages:
        return 1.0
    return 0.35


def _workflow_reward_components(
    *,
    profile: WorkflowProfile,
    stage_label: str,
    focus_kpi_id: str,
    active_bottleneck_ids: Sequence[str],
    interventions: Sequence[str],
) -> tuple[float, dict[str, float]]:
    if not interventions:
        return 0.0, {
            "pair_alignment": 0.0,
            "stage_fit": 0.0,
            "bottleneck_relief": 0.0,
            "sequence_synergy": 0.0,
            "coverage_breadth": 0.0,
        }
    primary = interventions[0]
    followup = interventions[1] if len(interventions) > 1 else ""
    if (focus_kpi_id, primary) in profile.primary_pairs:
        pair_alignment = 1.0
    elif primary in profile.default_improvements:
        pair_alignment = 0.72
    else:
        pair_alignment = 0.4
    stage_fit_values = [_workflow_stage_fit_score(improvement_id, stage_label) for improvement_id in interventions]
    stage_fit = sum(stage_fit_values) / max(len(stage_fit_values), 1)
    relieved_bottlenecks = {
        bottleneck_id
        for improvement_id in interventions
        for bottleneck_id in _WORKFLOW_IMPROVEMENT_BOTTLENECK_HINTS.get(improvement_id, ())
    }
    if active_bottleneck_ids:
        bottleneck_relief = len(relieved_bottlenecks.intersection(active_bottleneck_ids)) / float(
            len(set(active_bottleneck_ids))
        )
    else:
        bottleneck_relief = 0.5
    if not followup:
        sequence_synergy = 0.45
    elif followup in _WORKFLOW_IMPROVEMENT_FOLLOWUPS.get(primary, ()):
        sequence_synergy = 1.0
    elif followup != primary and followup in profile.default_improvements:
        sequence_synergy = 0.65
    else:
        sequence_synergy = 0.25
    coverage_breadth = min(len(set(interventions)) / 2.0, 1.0)
    components = {
        "pair_alignment": round(pair_alignment, 6),
        "stage_fit": round(stage_fit, 6),
        "bottleneck_relief": round(bottleneck_relief, 6),
        "sequence_synergy": round(sequence_synergy, 6),
        "coverage_breadth": round(coverage_breadth, 6),
    }
    score = min(
        1.0,
        0.4 * pair_alignment
        + 0.2 * stage_fit
        + 0.2 * bottleneck_relief
        + 0.15 * sequence_synergy
        + 0.05 * coverage_breadth,
    )
    return score, components


def _workflow_motif_for_plan(
    *,
    primary_improvement_id: str,
    active_bottleneck_ids: Sequence[str],
) -> tuple[str, str]:
    motif_id = _WORKFLOW_IMPROVEMENT_TO_MOTIF.get(primary_improvement_id, "")
    if not motif_id:
        for bottleneck_id in active_bottleneck_ids:
            motif_id = _WORKFLOW_BOTTLENECK_TO_MOTIF.get(bottleneck_id, "")
            if motif_id:
                break
    if not motif_id:
        motif_id = "shared_visibility_layer"
    motif_label = _workflow_motif_label_map().get(motif_id, motif_id.replace("_", " "))
    return motif_id, motif_label


def _workflow_intervention_plan(
    *,
    profile: WorkflowProfile,
    stage_label: str,
    kpi_ids: Sequence[str],
    bottleneck_ids: Sequence[str],
    improvement_ids: Sequence[str],
) -> WorkflowInterventionPlan | None:
    kpi_pool = _workflow_unique_ids(tuple(kpi_ids) + tuple(profile.default_kpis))
    improvement_pool = _workflow_unique_ids(tuple(improvement_ids) + tuple(profile.default_improvements))
    if not kpi_pool or not improvement_pool:
        return None
    active_bottlenecks = _workflow_unique_ids(tuple(bottleneck_ids) + tuple(profile.default_bottlenecks))[:4]
    best_plan: WorkflowInterventionPlan | None = None
    best_key: tuple[float, float, float] | None = None
    for focus_kpi_id in kpi_pool[:4]:
        for primary_improvement_id in improvement_pool[:6]:
            primary_score, primary_components = _workflow_reward_components(
                profile=profile,
                stage_label=stage_label,
                focus_kpi_id=focus_kpi_id,
                active_bottleneck_ids=active_bottlenecks,
                interventions=(primary_improvement_id,),
            )
            followup_pool = _workflow_unique_ids(
                _WORKFLOW_IMPROVEMENT_FOLLOWUPS.get(primary_improvement_id, ())
                + improvement_pool
            )
            followup_candidates = tuple(
                candidate_id
                for candidate_id in followup_pool
                if candidate_id and candidate_id != primary_improvement_id
            ) or (primary_improvement_id,)
            for followup_improvement_id in followup_candidates[:6]:
                final_score, final_components = _workflow_reward_components(
                    profile=profile,
                    stage_label=stage_label,
                    focus_kpi_id=focus_kpi_id,
                    active_bottleneck_ids=active_bottlenecks,
                    interventions=(primary_improvement_id, followup_improvement_id),
                )
                key = (
                    final_score,
                    primary_components.get("pair_alignment", 0.0),
                    primary_score,
                )
                if best_key is not None and key <= best_key:
                    continue
                primary_bucket_id, primary_bucket_label = _workflow_reward_bucket(primary_score)
                final_bucket_id, final_bucket_label = _workflow_reward_bucket(final_score)
                motif_id, motif_label = _workflow_motif_for_plan(
                    primary_improvement_id=primary_improvement_id,
                    active_bottleneck_ids=active_bottlenecks,
                )
                best_key = key
                best_plan = WorkflowInterventionPlan(
                    focus_kpi_id=focus_kpi_id,
                    primary_improvement_id=primary_improvement_id,
                    followup_improvement_id=followup_improvement_id,
                    active_bottleneck_ids=active_bottlenecks,
                    kpi_candidate_ids=kpi_pool[:4],
                    improvement_candidate_ids=improvement_pool[:6],
                    primary_reward_score=primary_score,
                    final_reward_score=final_score,
                    primary_reward_bucket_id=primary_bucket_id,
                    final_reward_bucket_id=final_bucket_id,
                    primary_reward_bucket_label=primary_bucket_label,
                    final_reward_bucket_label=final_bucket_label,
                    primary_reward_components=primary_components,
                    final_reward_components=final_components,
                    motif_id=motif_id,
                    motif_label=motif_label,
                )
    return best_plan


def _workflow_case_frames(
    records: Sequence[OscarScopeRecord],
) -> tuple[WorkflowCaseFrame, ...]:
    frames: list[WorkflowCaseFrame] = []
    for record in records:
        if record.view != "native_chunk":
            continue
        header, body = _split_scope_text(record.text)
        if not _workflow_body_is_usable(body):
            continue
        doc_title = str(record.metadata.get("doc_title", header.get("doc_title", record.document_id)))
        profile = _workflow_profile_for_document(doc_id=record.document_id, doc_title=doc_title)
        if profile is None:
            continue
        section_path = str(record.metadata.get("section_path", header.get("section_path", doc_title))).strip() or doc_title
        source_text = " ".join(part for part in (doc_title, section_path, body) if part)
        kpi_tags = _workflow_ranked_tags(
            source_text,
            entries=_WORKFLOW_KPI_RULES,
            defaults=profile.default_kpis,
            limit=4,
        )
        bottleneck_tags = _workflow_ranked_tags(
            source_text,
            entries=_WORKFLOW_BOTTLENECK_RULES,
            defaults=profile.default_bottlenecks,
            limit=4,
        )
        improvement_tags = _workflow_ranked_tags(
            source_text,
            entries=_WORKFLOW_IMPROVEMENT_RULES,
            defaults=profile.default_improvements,
            limit=6,
        )
        stage_label = _workflow_stage_label(profile, text=source_text)
        plan = _workflow_intervention_plan(
            profile=profile,
            stage_label=stage_label,
            kpi_ids=kpi_tags,
            bottleneck_ids=bottleneck_tags,
            improvement_ids=improvement_tags,
        )
        if plan is None:
            continue
        frames.append(
            WorkflowCaseFrame(
                frame_id=f"{record.document_id}:workflow_case:{record.chunk_index}",
                record=record,
                profile=profile,
                doc_title=doc_title,
                section_path=section_path,
                body=body,
                source_text=source_text,
                stage_label=stage_label,
                concept_tags=_concept_tags_for_text(source_text),
                kpi_tags=kpi_tags,
                bottleneck_tags=bottleneck_tags,
                improvement_tags=improvement_tags,
                plan=plan,
            )
        )
    return tuple(frames)


def _workflow_frame_similarity(
    source_frame: WorkflowCaseFrame,
    target_frame: WorkflowCaseFrame,
) -> tuple[int, int, int]:
    shared_bottlenecks = len(set(source_frame.plan.active_bottleneck_ids).intersection(target_frame.plan.active_bottleneck_ids))
    reward_bucket_match = int(
        source_frame.plan.primary_reward_bucket_label == target_frame.plan.primary_reward_bucket_label
    )
    stage_match = int(source_frame.stage_label == target_frame.stage_label)
    return (shared_bottlenecks, reward_bucket_match, stage_match)


def _workflow_stage_label(profile: WorkflowProfile, *, text: str) -> str:
    lowered = text.lower()
    for stage_label, keywords in profile.stage_rules:
        if any(keyword in lowered for keyword in keywords):
            return stage_label
    return "workflow_overview"


def _workflow_body_is_usable(body: str) -> bool:
    stripped = " ".join(body.split())
    if len(stripped) < 120:
        return False
    if stripped.count(" . .") >= 2:
        return False
    return True


def _workflow_common_targets(
    *,
    record: OscarScopeRecord,
    doc_title: str,
    section_path: str,
    body: str,
    profile: WorkflowProfile,
    graph_index: dict[str, OscarDocumentGraphEntry],
    family: str,
    task_kind: str,
    stage_label: str,
    concept_tags: Sequence[str],
) -> dict[str, object]:
    return _augment_task_metadata(
        {
            "family": family,
            "task_kind": task_kind,
            "doc_group": str(record.metadata.get("doc_group", "unknown")),
            "doc_title": doc_title,
            "doc_id": record.document_id,
            "source_view": record.view,
            "section_path": section_path,
            "section_depth": _section_depth(section_path or doc_title),
            "section_path_leaf": _section_path_leaf(section_path or doc_title),
            "section_path_label": section_path or doc_title,
            "concept_tags": list(concept_tags),
            "workflow_profile_id": profile.profile_id,
            "workflow_environment": profile.environment_label,
            "workflow_domain": profile.business_domain,
            "workflow_label": profile.workflow_label,
            "workflow_stage": stage_label,
            "excerpt_char_count": len(body),
        },
        graph_index=graph_index,
    )


def _workflow_environment_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> list[OscarScopeReasoningTask]:
    task_groups: dict[str, list[OscarScopeReasoningTask]] = {}
    environment_labels = [profile.environment_label for profile in _WORKFLOW_PROFILES]
    for record in records:
        if record.view != "native_chunk":
            continue
        header, body = _split_scope_text(record.text)
        if not _workflow_body_is_usable(body):
            continue
        doc_title = str(record.metadata.get("doc_title", header.get("doc_title", record.document_id)))
        profile = _workflow_profile_for_document(doc_id=record.document_id, doc_title=doc_title)
        if profile is None:
            continue
        section_path = str(record.metadata.get("section_path", header.get("section_path", doc_title))).strip() or doc_title
        source_text = " ".join(part for part in (doc_title, section_path, body) if part)
        concept_tags = _concept_tags_for_text(source_text)
        stage_label = _workflow_stage_label(profile, text=source_text)
        auxiliary_targets = _workflow_common_targets(
            record=record,
            doc_title=doc_title,
            section_path=section_path,
            body=body,
            profile=profile,
            graph_index=graph_index,
            family="oscar_workflow_environment",
            task_kind="workflow_environment",
            stage_label=stage_label,
            concept_tags=concept_tags,
        )
        auxiliary_targets = {
            **auxiliary_targets,
            "environment_candidates": list(environment_labels),
        }
        task_id = f"{record.document_id}:workflow_environment:{record.chunk_index}"
        text = _serialize_reasoning_task(
            task_id=task_id,
            family="oscar_workflow_environment",
            difficulty=2,
            trace_step="reason",
            answer_format="json",
            doc_title=doc_title,
            doc_group=str(record.metadata.get("doc_group", "unknown")),
            doc_id=record.document_id,
            source_kind=str(record.metadata.get("source_kind", "unknown")),
            source_view=record.view,
            section_path=section_path,
            question="Which real business workflow environment best matches this excerpt?",
            context_payload={
                "excerpt": body,
                "doc_title": doc_title,
                "workflow_stage": stage_label,
                "environment_candidates": environment_labels,
            },
            target_answer={"environment": profile.environment_label},
            auxiliary_targets=auxiliary_targets,
            task_kind="workflow_environment",
        )
        task_groups.setdefault(profile.profile_id, []).append(
            OscarScopeReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family="oscar_workflow_environment",
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return _interleave_task_groups(tuple(task_groups.values()))


def _workflow_multi_tag_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
    family: str,
    task_kind: str,
    question: str,
    entries: Sequence[WorkflowTaxonomyEntry],
    default_attr_name: str,
    output_key: str,
) -> list[OscarScopeReasoningTask]:
    task_groups: dict[str, list[OscarScopeReasoningTask]] = {}
    candidate_ids = [entry.tag_id for entry in entries]
    candidate_labels = [entry.label for entry in entries]
    label_map = _workflow_taxonomy_label_map(entries)
    for record in records:
        if record.view != "native_chunk":
            continue
        header, body = _split_scope_text(record.text)
        if not _workflow_body_is_usable(body):
            continue
        doc_title = str(record.metadata.get("doc_title", header.get("doc_title", record.document_id)))
        profile = _workflow_profile_for_document(doc_id=record.document_id, doc_title=doc_title)
        if profile is None:
            continue
        section_path = str(record.metadata.get("section_path", header.get("section_path", doc_title))).strip() or doc_title
        source_text = " ".join(part for part in (doc_title, section_path, body) if part)
        defaults = tuple(getattr(profile, default_attr_name))
        selected_tags = _workflow_ranked_tags(source_text, entries=entries, defaults=defaults, limit=4)
        if not selected_tags:
            continue
        concept_tags = _concept_tags_for_text(source_text)
        stage_label = _workflow_stage_label(profile, text=source_text)
        auxiliary_targets = _workflow_common_targets(
            record=record,
            doc_title=doc_title,
            section_path=section_path,
            body=body,
            profile=profile,
            graph_index=graph_index,
            family=family,
            task_kind=task_kind,
            stage_label=stage_label,
            concept_tags=concept_tags,
        )
        auxiliary_targets = {
            **auxiliary_targets,
            f"{output_key}_ids": list(selected_tags),
            f"{output_key}_labels": [label_map[tag_id] for tag_id in selected_tags],
            f"candidate_{output_key}_ids": list(candidate_ids),
            f"candidate_{output_key}_labels": list(candidate_labels),
        }
        task_id = f"{record.document_id}:{task_kind}:{record.chunk_index}"
        text = _serialize_reasoning_task(
            task_id=task_id,
            family=family,
            difficulty=min(5, max(1, len(selected_tags))),
            trace_step="reason",
            answer_format="json",
            doc_title=doc_title,
            doc_group=str(record.metadata.get("doc_group", "unknown")),
            doc_id=record.document_id,
            source_kind=str(record.metadata.get("source_kind", "unknown")),
            source_view=record.view,
            section_path=section_path,
            question=question,
            context_payload={
                "excerpt": body,
                "doc_title": doc_title,
                "workflow_environment": profile.environment_label,
                "workflow_stage": stage_label,
                f"candidate_{output_key}_labels": candidate_labels,
            },
            target_answer={output_key: list(selected_tags)},
            auxiliary_targets=auxiliary_targets,
            task_kind=task_kind,
        )
        task_groups.setdefault(profile.profile_id, []).append(
            OscarScopeReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family=family,
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return _interleave_task_groups(tuple(task_groups.values()))


def _workflow_kpi_improvement_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> list[OscarScopeReasoningTask]:
    task_groups: dict[str, list[OscarScopeReasoningTask]] = {}
    kpi_label_map = _workflow_taxonomy_label_map(_WORKFLOW_KPI_RULES)
    improvement_label_map = _workflow_taxonomy_label_map(_WORKFLOW_IMPROVEMENT_RULES)
    canonical_kpi_label_map = _workflow_canonical_kpi_label_map()
    canonical_intervention_label_map = _workflow_canonical_intervention_label_map()
    improvement_candidate_ids = [entry.tag_id for entry in _WORKFLOW_IMPROVEMENT_RULES]
    improvement_candidate_labels = [entry.label for entry in _WORKFLOW_IMPROVEMENT_RULES]
    for record in records:
        if record.view != "native_chunk":
            continue
        header, body = _split_scope_text(record.text)
        if not _workflow_body_is_usable(body):
            continue
        doc_title = str(record.metadata.get("doc_title", header.get("doc_title", record.document_id)))
        profile = _workflow_profile_for_document(doc_id=record.document_id, doc_title=doc_title)
        if profile is None:
            continue
        section_path = str(record.metadata.get("section_path", header.get("section_path", doc_title))).strip() or doc_title
        source_text = " ".join(part for part in (doc_title, section_path, body) if part)
        kpi_tags = _workflow_ranked_tags(source_text, entries=_WORKFLOW_KPI_RULES, defaults=profile.default_kpis, limit=4)
        bottleneck_tags = _workflow_ranked_tags(
            source_text,
            entries=_WORKFLOW_BOTTLENECK_RULES,
            defaults=profile.default_bottlenecks,
            limit=4,
        )
        improvement_tags = _workflow_ranked_tags(
            source_text,
            entries=_WORKFLOW_IMPROVEMENT_RULES,
            defaults=profile.default_improvements,
            limit=4,
        )
        selected_pairs = [
            (kpi_id, improvement_id)
            for kpi_id, improvement_id in profile.primary_pairs
            if kpi_id in kpi_tags and improvement_id in improvement_tags
        ]
        if not selected_pairs and kpi_tags and improvement_tags:
            selected_pairs = [(kpi_tags[0], improvement_tags[0])]
        if not selected_pairs:
            continue
        concept_tags = _concept_tags_for_text(source_text)
        stage_label = _workflow_stage_label(profile, text=source_text)
        for pair_index, (kpi_id, improvement_id) in enumerate(selected_pairs[:2]):
            motif_id, motif_label = _workflow_motif_for_plan(
                primary_improvement_id=improvement_id,
                active_bottleneck_ids=bottleneck_tags,
            )
            canonical_kpi_id = _workflow_canonical_kpi_id(kpi_id)
            canonical_improvement_id = _workflow_canonical_intervention_id(improvement_id)
            auxiliary_targets = _workflow_common_targets(
                record=record,
                doc_title=doc_title,
                section_path=section_path,
                body=body,
                profile=profile,
                graph_index=graph_index,
                family="oscar_workflow_kpi_improvement",
                task_kind="workflow_kpi_improvement",
                stage_label=stage_label,
                concept_tags=concept_tags,
            )
            auxiliary_targets = {
                **auxiliary_targets,
                "focus_kpi_id": kpi_id,
                "focus_kpi_label": kpi_label_map[kpi_id],
                "focus_improvement_id": improvement_id,
                "focus_improvement_label": improvement_label_map[improvement_id],
                "workflow_canonical_kpi_id": canonical_kpi_id,
                "workflow_canonical_kpi_label": canonical_kpi_label_map[canonical_kpi_id],
                "workflow_canonical_improvement_id": canonical_improvement_id,
                "workflow_canonical_improvement_label": canonical_intervention_label_map[canonical_improvement_id],
                "workflow_motif_id": motif_id,
                "workflow_motif_label": motif_label,
                "candidate_improvement_ids": list(improvement_candidate_ids),
                "candidate_improvement_labels": list(improvement_candidate_labels),
                "available_kpi_ids": list(kpi_tags),
                "available_improvement_ids": list(improvement_tags),
            }
            task_id = f"{record.document_id}:workflow_kpi_improvement:{record.chunk_index}:{pair_index}"
            text = _serialize_reasoning_task(
                task_id=task_id,
                family="oscar_workflow_kpi_improvement",
                difficulty=4,
                trace_step="reason",
                answer_format="json",
                doc_title=doc_title,
                doc_group=str(record.metadata.get("doc_group", "unknown")),
                doc_id=record.document_id,
                source_kind=str(record.metadata.get("source_kind", "unknown")),
                source_view=record.view,
                section_path=section_path,
                question=f"Which improvement lever most directly moves KPI {kpi_label_map[kpi_id]} in this workflow excerpt?",
                context_payload={
                    "excerpt": body,
                    "doc_title": doc_title,
                    "workflow_environment": profile.environment_label,
                    "workflow_stage": stage_label,
                    "focus_kpi": {
                        "id": kpi_id,
                        "label": kpi_label_map[kpi_id],
                    },
                    "candidate_improvements": improvement_candidate_labels,
                },
                target_answer={
                    "kpi": kpi_id,
                    "improvement": improvement_id,
                },
                auxiliary_targets=auxiliary_targets,
                task_kind="workflow_kpi_improvement",
            )
            task_groups.setdefault(profile.profile_id, []).append(
                OscarScopeReasoningTask(
                    task_id=task_id,
                    benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                    family="oscar_workflow_kpi_improvement",
                    trace_step="reason",
                    text=text,
                    metadata=auxiliary_targets,
                )
            )
    return _interleave_task_groups(tuple(task_groups.values()))


def _workflow_intervention_trace_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> list[OscarScopeReasoningTask]:
    task_groups: dict[str, list[list[OscarScopeReasoningTask]]] = {}
    kpi_label_map = _workflow_taxonomy_label_map(_WORKFLOW_KPI_RULES)
    bottleneck_label_map = _workflow_taxonomy_label_map(_WORKFLOW_BOTTLENECK_RULES)
    improvement_label_map = _workflow_taxonomy_label_map(_WORKFLOW_IMPROVEMENT_RULES)
    canonical_kpi_label_map = _workflow_canonical_kpi_label_map()
    canonical_intervention_label_map = _workflow_canonical_intervention_label_map()
    for record in records:
        if record.view != "native_chunk":
            continue
        header, body = _split_scope_text(record.text)
        if not _workflow_body_is_usable(body):
            continue
        doc_title = str(record.metadata.get("doc_title", header.get("doc_title", record.document_id)))
        profile = _workflow_profile_for_document(doc_id=record.document_id, doc_title=doc_title)
        if profile is None:
            continue
        section_path = str(record.metadata.get("section_path", header.get("section_path", doc_title))).strip() or doc_title
        source_text = " ".join(part for part in (doc_title, section_path, body) if part)
        kpi_tags = _workflow_ranked_tags(source_text, entries=_WORKFLOW_KPI_RULES, defaults=profile.default_kpis, limit=4)
        bottleneck_tags = _workflow_ranked_tags(
            source_text,
            entries=_WORKFLOW_BOTTLENECK_RULES,
            defaults=profile.default_bottlenecks,
            limit=4,
        )
        improvement_tags = _workflow_ranked_tags(
            source_text,
            entries=_WORKFLOW_IMPROVEMENT_RULES,
            defaults=profile.default_improvements,
            limit=6,
        )
        concept_tags = _concept_tags_for_text(source_text)
        stage_label = _workflow_stage_label(profile, text=source_text)
        plan = _workflow_intervention_plan(
            profile=profile,
            stage_label=stage_label,
            kpi_ids=kpi_tags,
            bottleneck_ids=bottleneck_tags,
            improvement_ids=improvement_tags,
        )
        if plan is None:
            continue

        focus_kpi_label = kpi_label_map[plan.focus_kpi_id]
        primary_improvement_label = improvement_label_map[plan.primary_improvement_id]
        followup_improvement_label = improvement_label_map[plan.followup_improvement_id]
        canonical_kpi_id = _workflow_canonical_kpi_id(plan.focus_kpi_id)
        canonical_kpi_label = canonical_kpi_label_map[canonical_kpi_id]
        canonical_primary_improvement_id = _workflow_canonical_intervention_id(plan.primary_improvement_id)
        canonical_primary_improvement_label = canonical_intervention_label_map[canonical_primary_improvement_id]
        canonical_followup_improvement_id = _workflow_canonical_intervention_id(plan.followup_improvement_id)
        canonical_followup_improvement_label = canonical_intervention_label_map[canonical_followup_improvement_id]
        candidate_kpi_labels = [kpi_label_map[kpi_id] for kpi_id in plan.kpi_candidate_ids]
        candidate_improvement_labels = [
            improvement_label_map[improvement_id]
            for improvement_id in plan.improvement_candidate_ids
        ]
        active_bottleneck_labels = [
            bottleneck_label_map[bottleneck_id]
            for bottleneck_id in plan.active_bottleneck_ids
            if bottleneck_id in bottleneck_label_map
        ]
        trajectory_id = f"{record.document_id}:workflow_intervention_trace:{record.chunk_index}"
        trace_length = 3
        previous_action = "<start>"
        previous_action_name = "<start>"

        common_targets = _workflow_common_targets(
            record=record,
            doc_title=doc_title,
            section_path=section_path,
            body=body,
            profile=profile,
            graph_index=graph_index,
            family="oscar_workflow_intervention_trace",
            task_kind="workflow_intervention_trace",
            stage_label=stage_label,
            concept_tags=concept_tags,
        )

        step0_targets = {
            **common_targets,
            "target_action_name": "select_kpi_family",
            "focus_kpi_id": plan.focus_kpi_id,
            "focus_kpi_label": focus_kpi_label,
            "workflow_canonical_kpi_id": canonical_kpi_id,
            "workflow_canonical_kpi_label": canonical_kpi_label,
            "candidate_kpi_ids": list(plan.kpi_candidate_ids),
            "candidate_kpi_labels": candidate_kpi_labels,
            "active_bottleneck_ids": list(plan.active_bottleneck_ids),
            "active_bottleneck_labels": active_bottleneck_labels,
            "workflow_motif_id": plan.motif_id,
            "workflow_motif_label": plan.motif_label,
            "workflow_reward_score": round(plan.final_reward_score, 6),
            "workflow_reward_bucket_id": plan.final_reward_bucket_id,
            "workflow_reward_bucket_label": plan.final_reward_bucket_label,
            "workflow_reward_components": plan.final_reward_components,
            "workflow_selected_improvement_ids": [
                plan.primary_improvement_id,
                plan.followup_improvement_id,
            ],
            "workflow_canonical_selected_improvement_ids": [
                canonical_primary_improvement_id,
                canonical_followup_improvement_id,
            ],
            "workflow_canonical_selected_improvement_labels": [
                canonical_primary_improvement_label,
                canonical_followup_improvement_label,
            ],
            "workflow_selected_improvement_labels": [
                primary_improvement_label,
                followup_improvement_label,
            ],
            "trajectory_id": trajectory_id,
            "step_index": 0,
            "trace_length": trace_length,
        }
        step0_action = {
            "name": "select_kpi_family",
            "action": {
                "kpi_family_id": canonical_kpi_id,
            },
        }
        step0_text = _serialize_workflow_decision_step(
            task_id=trajectory_id,
            family="oscar_workflow_intervention_trace",
            difficulty=5,
            trace_step="select_focus_kpi",
            step_index=0,
            trace_length=trace_length,
            previous_action=previous_action,
            candidate_bucket=(
                "family:oscar_workflow_intervention_trace"
                f"|motif:{plan.motif_id}|kpi_family:{canonical_kpi_id}|step:0|prev:<start>"
            ),
            doc_title=doc_title,
            doc_group=str(record.metadata.get("doc_group", "unknown")),
            doc_id=record.document_id,
            source_kind=str(record.metadata.get("source_kind", "unknown")),
            source_view=record.view,
            section_path=section_path,
            question="Which KPI should this workflow intervention sequence optimize first?",
            context_payload={
                "excerpt": body,
                "doc_title": doc_title,
                "workflow_environment": profile.environment_label,
                "workflow_stage": stage_label,
                "workflow_motif": {
                    "id": plan.motif_id,
                    "label": plan.motif_label,
                },
                "candidate_kpis": [
                    {
                        "id": kpi_id,
                        "label": kpi_label_map[kpi_id],
                        "canonical_id": _workflow_canonical_kpi_id(kpi_id),
                        "canonical_label": canonical_kpi_label_map[_workflow_canonical_kpi_id(kpi_id)],
                    }
                    for kpi_id in plan.kpi_candidate_ids
                ],
                "active_bottlenecks": active_bottleneck_labels,
            },
            verifier_context={
                "projected_reward_score": round(plan.final_reward_score, 6),
                "projected_reward_bucket": plan.final_reward_bucket_label,
                "pending_interventions": 2,
                "should_stop": False,
            },
            auxiliary_targets=step0_targets,
            target_action=step0_action,
            task_kind="workflow_intervention_trace",
            state_tokens=(
                *_state_tokens(
                    doc_group=str(record.metadata.get("doc_group", "unknown")),
                    source_kind=str(record.metadata.get("source_kind", "unknown")),
                    source_view=record.view,
                    family="oscar_workflow_intervention_trace",
                    section_depth=_section_depth(section_path or doc_title),
                    task_kind="workflow_intervention_trace",
                ),
                f"workflow.profile={profile.profile_id}",
                f"workflow.stage={stage_label}",
                f"workflow.motif={plan.motif_id}",
                "executor.step=select_focus_kpi",
            ),
            state_scalars=(
                0.0,
                float(len(plan.active_bottleneck_ids)),
                plan.final_reward_score,
            ),
        )
        trajectory_tasks = [
            OscarScopeReasoningTask(
                task_id=f"{trajectory_id}:step_0",
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family="oscar_workflow_intervention_trace",
                trace_step="select_focus_kpi",
                text=step0_text,
                metadata=step0_targets,
            )
        ]
        previous_action = json.dumps(step0_action, separators=(",", ":"), sort_keys=True)
        previous_action_name = "select_kpi_family"

        step1_targets = {
            **common_targets,
            "target_action_name": "select_intervention_family",
            "focus_kpi_id": plan.focus_kpi_id,
            "focus_kpi_label": focus_kpi_label,
            "workflow_canonical_kpi_id": canonical_kpi_id,
            "workflow_canonical_kpi_label": canonical_kpi_label,
            "focus_improvement_id": plan.primary_improvement_id,
            "focus_improvement_label": primary_improvement_label,
            "primary_improvement_id": plan.primary_improvement_id,
            "primary_improvement_label": primary_improvement_label,
            "workflow_canonical_primary_improvement_id": canonical_primary_improvement_id,
            "workflow_canonical_primary_improvement_label": canonical_primary_improvement_label,
            "followup_improvement_id": plan.followup_improvement_id,
            "followup_improvement_label": followup_improvement_label,
            "workflow_canonical_followup_improvement_id": canonical_followup_improvement_id,
            "workflow_canonical_followup_improvement_label": canonical_followup_improvement_label,
            "candidate_improvement_ids": list(plan.improvement_candidate_ids),
            "candidate_improvement_labels": candidate_improvement_labels,
            "active_bottleneck_ids": list(plan.active_bottleneck_ids),
            "active_bottleneck_labels": active_bottleneck_labels,
            "workflow_motif_id": plan.motif_id,
            "workflow_motif_label": plan.motif_label,
            "workflow_reward_score": round(plan.primary_reward_score, 6),
            "workflow_reward_bucket_id": plan.primary_reward_bucket_id,
            "workflow_reward_bucket_label": plan.primary_reward_bucket_label,
            "workflow_reward_components": plan.primary_reward_components,
            "workflow_selected_improvement_ids": [plan.primary_improvement_id],
            "workflow_canonical_selected_improvement_ids": [canonical_primary_improvement_id],
            "workflow_canonical_selected_improvement_labels": [canonical_primary_improvement_label],
            "workflow_selected_improvement_labels": [primary_improvement_label],
            "trajectory_id": trajectory_id,
            "step_index": 1,
            "trace_length": trace_length,
        }
        step1_action = {
            "name": "select_intervention_family",
            "action": {
                "intervention_family_id": canonical_primary_improvement_id,
            },
        }
        step1_text = _serialize_workflow_decision_step(
            task_id=trajectory_id,
            family="oscar_workflow_intervention_trace",
            difficulty=5,
            trace_step="apply_primary_intervention",
            step_index=1,
            trace_length=trace_length,
            previous_action=previous_action,
            candidate_bucket=(
                "family:oscar_workflow_intervention_trace"
                f"|motif:{plan.motif_id}|kpi_family:{canonical_kpi_id}"
                f"|step:1|prev:{previous_action_name}"
            ),
            doc_title=doc_title,
            doc_group=str(record.metadata.get("doc_group", "unknown")),
            doc_id=record.document_id,
            source_kind=str(record.metadata.get("source_kind", "unknown")),
            source_view=record.view,
            section_path=section_path,
            question=f"Which primary intervention should move KPI {focus_kpi_label} first in this workflow excerpt?",
            context_payload={
                "excerpt": body,
                "doc_title": doc_title,
                "workflow_environment": profile.environment_label,
                "workflow_stage": stage_label,
                "focus_kpi": {
                    "id": plan.focus_kpi_id,
                    "label": focus_kpi_label,
                    "canonical_id": canonical_kpi_id,
                    "canonical_label": canonical_kpi_label,
                },
                "active_bottlenecks": active_bottleneck_labels,
                "candidate_improvements": [
                    {
                        "id": improvement_id,
                        "label": improvement_label_map[improvement_id],
                        "canonical_id": _workflow_canonical_intervention_id(improvement_id),
                        "canonical_label": canonical_intervention_label_map[_workflow_canonical_intervention_id(improvement_id)],
                    }
                    for improvement_id in plan.improvement_candidate_ids
                ],
            },
            verifier_context={
                "projected_primary_reward_score": round(plan.primary_reward_score, 6),
                "projected_primary_reward_bucket": plan.primary_reward_bucket_label,
                "projected_final_reward_score": round(plan.final_reward_score, 6),
                "pending_interventions": 1,
                "should_stop": False,
            },
            auxiliary_targets=step1_targets,
            target_action=step1_action,
            task_kind="workflow_intervention_trace",
            state_tokens=(
                *_state_tokens(
                    doc_group=str(record.metadata.get("doc_group", "unknown")),
                    source_kind=str(record.metadata.get("source_kind", "unknown")),
                    source_view=record.view,
                    family="oscar_workflow_intervention_trace",
                    section_depth=_section_depth(section_path or doc_title),
                    task_kind="workflow_intervention_trace",
                ),
                f"workflow.profile={profile.profile_id}",
                f"workflow.stage={stage_label}",
                f"workflow.motif={plan.motif_id}",
                f"selected.kpi_family={canonical_kpi_id}",
                "executor.step=apply_primary_intervention",
            ),
            state_scalars=(
                0.5,
                float(len(plan.active_bottleneck_ids)),
                plan.primary_reward_score,
            ),
        )
        trajectory_tasks.append(
            OscarScopeReasoningTask(
                task_id=f"{trajectory_id}:step_1",
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family="oscar_workflow_intervention_trace",
                trace_step="apply_primary_intervention",
                text=step1_text,
                metadata=step1_targets,
            )
        )
        previous_action = json.dumps(step1_action, separators=(",", ":"), sort_keys=True)
        previous_action_name = "select_intervention_family"

        step2_targets = {
            **common_targets,
            "target_action_name": "select_intervention_family",
            "focus_kpi_id": plan.focus_kpi_id,
            "focus_kpi_label": focus_kpi_label,
            "workflow_canonical_kpi_id": canonical_kpi_id,
            "workflow_canonical_kpi_label": canonical_kpi_label,
            "focus_improvement_id": plan.followup_improvement_id,
            "focus_improvement_label": followup_improvement_label,
            "primary_improvement_id": plan.primary_improvement_id,
            "primary_improvement_label": primary_improvement_label,
            "workflow_canonical_primary_improvement_id": canonical_primary_improvement_id,
            "workflow_canonical_primary_improvement_label": canonical_primary_improvement_label,
            "followup_improvement_id": plan.followup_improvement_id,
            "followup_improvement_label": followup_improvement_label,
            "workflow_canonical_followup_improvement_id": canonical_followup_improvement_id,
            "workflow_canonical_followup_improvement_label": canonical_followup_improvement_label,
            "candidate_improvement_ids": list(plan.improvement_candidate_ids),
            "candidate_improvement_labels": candidate_improvement_labels,
            "active_bottleneck_ids": list(plan.active_bottleneck_ids),
            "active_bottleneck_labels": active_bottleneck_labels,
            "workflow_motif_id": plan.motif_id,
            "workflow_motif_label": plan.motif_label,
            "workflow_reward_score": round(plan.final_reward_score, 6),
            "workflow_reward_bucket_id": plan.final_reward_bucket_id,
            "workflow_reward_bucket_label": plan.final_reward_bucket_label,
            "workflow_reward_components": plan.final_reward_components,
            "workflow_selected_improvement_ids": [
                plan.primary_improvement_id,
                plan.followup_improvement_id,
            ],
            "workflow_canonical_selected_improvement_ids": [
                canonical_primary_improvement_id,
                canonical_followup_improvement_id,
            ],
            "workflow_canonical_selected_improvement_labels": [
                canonical_primary_improvement_label,
                canonical_followup_improvement_label,
            ],
            "workflow_selected_improvement_labels": [
                primary_improvement_label,
                followup_improvement_label,
            ],
            "trajectory_id": trajectory_id,
            "step_index": 2,
            "trace_length": trace_length,
        }
        step2_action = {
            "name": "select_intervention_family",
            "action": {
                "intervention_family_id": canonical_followup_improvement_id,
            },
        }
        step2_text = _serialize_workflow_decision_step(
            task_id=trajectory_id,
            family="oscar_workflow_intervention_trace",
            difficulty=5,
            trace_step="apply_followup_intervention",
            step_index=2,
            trace_length=trace_length,
            previous_action=previous_action,
            candidate_bucket=(
                "family:oscar_workflow_intervention_trace"
                f"|motif:{plan.motif_id}|kpi_family:{canonical_kpi_id}"
                f"|primary_family:{canonical_primary_improvement_id}"
                f"|step:2|prev:{previous_action_name}"
            ),
            doc_title=doc_title,
            doc_group=str(record.metadata.get("doc_group", "unknown")),
            doc_id=record.document_id,
            source_kind=str(record.metadata.get("source_kind", "unknown")),
            source_view=record.view,
            section_path=section_path,
            question=(
                f"Which follow-up intervention best compounds KPI {focus_kpi_label} "
                f"after {primary_improvement_label}?"
            ),
            context_payload={
                "excerpt": body,
                "doc_title": doc_title,
                "workflow_environment": profile.environment_label,
                "workflow_stage": stage_label,
                "focus_kpi": {
                    "id": plan.focus_kpi_id,
                    "label": focus_kpi_label,
                    "canonical_id": canonical_kpi_id,
                    "canonical_label": canonical_kpi_label,
                },
                "active_bottlenecks": active_bottleneck_labels,
                "primary_intervention": {
                    "id": plan.primary_improvement_id,
                    "label": primary_improvement_label,
                    "canonical_id": canonical_primary_improvement_id,
                    "canonical_label": canonical_primary_improvement_label,
                },
                "candidate_improvements": [
                    {
                        "id": improvement_id,
                        "label": improvement_label_map[improvement_id],
                        "canonical_id": _workflow_canonical_intervention_id(improvement_id),
                        "canonical_label": canonical_intervention_label_map[_workflow_canonical_intervention_id(improvement_id)],
                    }
                    for improvement_id in plan.improvement_candidate_ids
                    if improvement_id != plan.primary_improvement_id
                ],
            },
            verifier_context={
                "projected_reward_score": round(plan.final_reward_score, 6),
                "projected_reward_bucket": plan.final_reward_bucket_label,
                "reward_gain_vs_primary": round(plan.final_reward_score - plan.primary_reward_score, 6),
                "reward_components": plan.final_reward_components,
                "pending_interventions": 0,
                "should_stop": True,
            },
            auxiliary_targets=step2_targets,
            target_action=step2_action,
            task_kind="workflow_intervention_trace",
            state_tokens=(
                *_state_tokens(
                    doc_group=str(record.metadata.get("doc_group", "unknown")),
                    source_kind=str(record.metadata.get("source_kind", "unknown")),
                    source_view=record.view,
                    family="oscar_workflow_intervention_trace",
                    section_depth=_section_depth(section_path or doc_title),
                    task_kind="workflow_intervention_trace",
                ),
                f"workflow.profile={profile.profile_id}",
                f"workflow.stage={stage_label}",
                f"workflow.motif={plan.motif_id}",
                f"selected.kpi_family={canonical_kpi_id}",
                f"selected.primary_intervention_family={canonical_primary_improvement_id}",
                "executor.step=apply_followup_intervention",
            ),
            state_scalars=(
                1.0,
                float(len(plan.active_bottleneck_ids)),
                plan.final_reward_score,
            ),
        )
        trajectory_tasks.append(
            OscarScopeReasoningTask(
                task_id=f"{trajectory_id}:step_2",
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family="oscar_workflow_intervention_trace",
                trace_step="apply_followup_intervention",
                text=step2_text,
                metadata=step2_targets,
            )
        )
        task_groups.setdefault(profile.profile_id, []).append(trajectory_tasks)
    ordered_profiles = [list(group) for group in task_groups.values() if group]
    if not ordered_profiles:
        return []
    selected: list[OscarScopeReasoningTask] = []
    offsets = [0 for _ in ordered_profiles]
    while True:
        progressed = False
        for group_index, trajectory_group in enumerate(ordered_profiles):
            offset = offsets[group_index]
            if offset >= len(trajectory_group):
                continue
            selected.extend(trajectory_group[offset])
            offsets[group_index] += 1
            progressed = True
        if not progressed:
            break
    return selected


def _workflow_case_analogy_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> list[OscarScopeReasoningTask]:
    task_groups: dict[str, list[OscarScopeReasoningTask]] = {}
    frames = _workflow_case_frames(records)
    motif_label_map = _workflow_motif_label_map()
    kpi_label_map = _workflow_taxonomy_label_map(_WORKFLOW_KPI_RULES)
    improvement_label_map = _workflow_taxonomy_label_map(_WORKFLOW_IMPROVEMENT_RULES)
    canonical_kpi_label_map = _workflow_canonical_kpi_label_map()
    canonical_intervention_label_map = _workflow_canonical_intervention_label_map()
    all_motif_ids = list(OSCAR_WORKFLOW_MOTIF_IDS)
    all_motif_labels = [motif_label_map[motif_id] for motif_id in all_motif_ids]
    for source_frame in frames:
        candidate_targets = [
            target_frame
            for target_frame in frames
            if target_frame.profile.profile_id != source_frame.profile.profile_id
            and target_frame.plan.motif_id == source_frame.plan.motif_id
        ]
        if not candidate_targets:
            continue
        candidate_targets.sort(
            key=lambda target_frame: (
                *_workflow_frame_similarity(source_frame, target_frame),
                target_frame.profile.profile_id,
                target_frame.frame_id,
            ),
            reverse=True,
        )
        target_frame = candidate_targets[0]
        auxiliary_targets = _workflow_common_targets(
            record=target_frame.record,
            doc_title=target_frame.doc_title,
            section_path=target_frame.section_path,
            body=target_frame.body,
            profile=target_frame.profile,
            graph_index=graph_index,
            family="oscar_workflow_case_analogy",
            task_kind="workflow_case_analogy",
            stage_label=target_frame.stage_label,
            concept_tags=target_frame.concept_tags,
        )
        auxiliary_targets = {
            **auxiliary_targets,
            "workflow_motif_id": source_frame.plan.motif_id,
            "workflow_motif_label": source_frame.plan.motif_label,
            "source_workflow_canonical_kpi_id": _workflow_canonical_kpi_id(source_frame.plan.focus_kpi_id),
            "source_workflow_canonical_kpi_label": canonical_kpi_label_map[_workflow_canonical_kpi_id(source_frame.plan.focus_kpi_id)],
            "source_workflow_canonical_improvement_id": _workflow_canonical_intervention_id(source_frame.plan.primary_improvement_id),
            "source_workflow_canonical_improvement_label": canonical_intervention_label_map[_workflow_canonical_intervention_id(source_frame.plan.primary_improvement_id)],
            "candidate_motif_ids": list(all_motif_ids),
            "candidate_motif_labels": list(all_motif_labels),
            "source_profile_id": source_frame.profile.profile_id,
            "source_workflow_environment": source_frame.profile.environment_label,
            "source_doc_id": source_frame.record.document_id,
            "source_doc_title": source_frame.doc_title,
            "source_focus_kpi_id": source_frame.plan.focus_kpi_id,
            "source_focus_kpi_label": kpi_label_map[source_frame.plan.focus_kpi_id],
            "source_focus_improvement_id": source_frame.plan.primary_improvement_id,
            "source_focus_improvement_label": improvement_label_map[source_frame.plan.primary_improvement_id],
            "source_workflow_reward_score": round(source_frame.plan.primary_reward_score, 6),
            "source_workflow_reward_bucket_label": source_frame.plan.primary_reward_bucket_label,
            "target_profile_id": target_frame.profile.profile_id,
            "target_workflow_environment": target_frame.profile.environment_label,
            "focus_kpi_id": target_frame.plan.focus_kpi_id,
            "focus_kpi_label": kpi_label_map[target_frame.plan.focus_kpi_id],
            "focus_improvement_id": target_frame.plan.primary_improvement_id,
            "focus_improvement_label": improvement_label_map[target_frame.plan.primary_improvement_id],
            "workflow_canonical_kpi_id": _workflow_canonical_kpi_id(target_frame.plan.focus_kpi_id),
            "workflow_canonical_kpi_label": canonical_kpi_label_map[_workflow_canonical_kpi_id(target_frame.plan.focus_kpi_id)],
            "workflow_canonical_improvement_id": _workflow_canonical_intervention_id(target_frame.plan.primary_improvement_id),
            "workflow_canonical_improvement_label": canonical_intervention_label_map[_workflow_canonical_intervention_id(target_frame.plan.primary_improvement_id)],
            "workflow_reward_score": round(target_frame.plan.primary_reward_score, 6),
            "workflow_reward_bucket_id": target_frame.plan.primary_reward_bucket_id,
            "workflow_reward_bucket_label": target_frame.plan.primary_reward_bucket_label,
        }
        task_id = (
            f"{source_frame.record.document_id}:workflow_case_analogy:{source_frame.record.chunk_index}:"
            f"{target_frame.record.document_id}:{target_frame.record.chunk_index}"
        )
        text = _serialize_reasoning_task(
            task_id=task_id,
            family="oscar_workflow_case_analogy",
            difficulty=5,
            trace_step="reason",
            answer_format="json",
            doc_title=target_frame.doc_title,
            doc_group=str(target_frame.record.metadata.get("doc_group", "unknown")),
            doc_id=target_frame.record.document_id,
            source_kind=str(target_frame.record.metadata.get("source_kind", "unknown")),
            source_view=target_frame.record.view,
            section_path=target_frame.section_path,
            question="What abstract workflow motif best links the source case and the target case?",
            context_payload={
                "source_case": {
                    "environment": source_frame.profile.environment_label,
                    "excerpt": source_frame.body,
                    "focus_kpi": {
                        "id": source_frame.plan.focus_kpi_id,
                        "label": kpi_label_map[source_frame.plan.focus_kpi_id],
                    },
                    "primary_intervention": {
                        "id": source_frame.plan.primary_improvement_id,
                        "label": improvement_label_map[source_frame.plan.primary_improvement_id],
                    },
                    "reward_bucket": source_frame.plan.primary_reward_bucket_label,
                },
                "target_case": {
                    "environment": target_frame.profile.environment_label,
                    "excerpt": target_frame.body,
                    "focus_kpi": {
                        "id": target_frame.plan.focus_kpi_id,
                        "label": kpi_label_map[target_frame.plan.focus_kpi_id],
                    },
                    "active_bottlenecks": list(target_frame.plan.active_bottleneck_ids),
                },
                "candidate_motifs": [
                    {"id": motif_id, "label": motif_label_map[motif_id]}
                    for motif_id in all_motif_ids
                ],
            },
            target_answer={
                "motif": source_frame.plan.motif_id,
                "motif_label": source_frame.plan.motif_label,
            },
            auxiliary_targets=auxiliary_targets,
            task_kind="workflow_case_analogy",
        )
        task_groups.setdefault(source_frame.plan.motif_id, []).append(
            OscarScopeReasoningTask(
                task_id=task_id,
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family="oscar_workflow_case_analogy",
                trace_step="reason",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return _interleave_task_groups(tuple(task_groups.values()))


def _workflow_transfer_tasks(
    records: Sequence[OscarScopeRecord],
    *,
    graph_index: dict[str, OscarDocumentGraphEntry],
) -> list[OscarScopeReasoningTask]:
    task_groups: dict[str, list[OscarScopeReasoningTask]] = {}
    frames = _workflow_case_frames(records)
    kpi_label_map = _workflow_taxonomy_label_map(_WORKFLOW_KPI_RULES)
    improvement_label_map = _workflow_taxonomy_label_map(_WORKFLOW_IMPROVEMENT_RULES)
    motif_label_map = _workflow_motif_label_map()
    canonical_kpi_label_map = _workflow_canonical_kpi_label_map()
    canonical_intervention_label_map = _workflow_canonical_intervention_label_map()
    for source_frame in frames:
        candidate_targets = [
            target_frame
            for target_frame in frames
            if target_frame.profile.profile_id != source_frame.profile.profile_id
            and target_frame.plan.motif_id == source_frame.plan.motif_id
        ]
        if not candidate_targets:
            continue
        candidate_targets.sort(
            key=lambda target_frame: (
                *_workflow_frame_similarity(source_frame, target_frame),
                target_frame.profile.profile_id,
                target_frame.frame_id,
            ),
            reverse=True,
        )
        target_frame = candidate_targets[0]
        trajectory_id = (
            f"{source_frame.record.document_id}:workflow_transfer:{source_frame.record.chunk_index}:"
            f"{target_frame.record.document_id}:{target_frame.record.chunk_index}"
        )
        source_canonical_kpi_id = _workflow_canonical_kpi_id(source_frame.plan.focus_kpi_id)
        target_canonical_kpi_id = _workflow_canonical_kpi_id(target_frame.plan.focus_kpi_id)
        source_canonical_improvement_id = _workflow_canonical_intervention_id(source_frame.plan.primary_improvement_id)
        target_canonical_improvement_id = _workflow_canonical_intervention_id(target_frame.plan.primary_improvement_id)
        auxiliary_targets = _workflow_common_targets(
            record=target_frame.record,
            doc_title=target_frame.doc_title,
            section_path=target_frame.section_path,
            body=target_frame.body,
            profile=target_frame.profile,
            graph_index=graph_index,
            family="oscar_workflow_transfer",
            task_kind="workflow_transfer",
            stage_label=target_frame.stage_label,
            concept_tags=target_frame.concept_tags,
        )
        auxiliary_targets = {
            **auxiliary_targets,
            "workflow_motif_id": source_frame.plan.motif_id,
            "workflow_motif_label": source_frame.plan.motif_label,
            "source_profile_id": source_frame.profile.profile_id,
            "source_workflow_environment": source_frame.profile.environment_label,
            "source_doc_id": source_frame.record.document_id,
            "source_doc_title": source_frame.doc_title,
            "source_focus_kpi_id": source_frame.plan.focus_kpi_id,
            "source_focus_kpi_label": kpi_label_map[source_frame.plan.focus_kpi_id],
            "source_focus_improvement_id": source_frame.plan.primary_improvement_id,
            "source_focus_improvement_label": improvement_label_map[source_frame.plan.primary_improvement_id],
            "source_workflow_canonical_kpi_id": source_canonical_kpi_id,
            "source_workflow_canonical_kpi_label": canonical_kpi_label_map[source_canonical_kpi_id],
            "source_workflow_canonical_improvement_id": source_canonical_improvement_id,
            "source_workflow_canonical_improvement_label": canonical_intervention_label_map[source_canonical_improvement_id],
            "source_followup_improvement_id": source_frame.plan.followup_improvement_id,
            "source_followup_improvement_label": improvement_label_map[source_frame.plan.followup_improvement_id],
            "source_workflow_reward_score": round(source_frame.plan.primary_reward_score, 6),
            "source_workflow_reward_bucket_label": source_frame.plan.primary_reward_bucket_label,
            "target_profile_id": target_frame.profile.profile_id,
            "target_workflow_environment": target_frame.profile.environment_label,
            "focus_kpi_id": target_frame.plan.focus_kpi_id,
            "focus_kpi_label": kpi_label_map[target_frame.plan.focus_kpi_id],
            "focus_improvement_id": target_frame.plan.primary_improvement_id,
            "focus_improvement_label": improvement_label_map[target_frame.plan.primary_improvement_id],
            "workflow_canonical_kpi_id": target_canonical_kpi_id,
            "workflow_canonical_kpi_label": canonical_kpi_label_map[target_canonical_kpi_id],
            "workflow_canonical_improvement_id": target_canonical_improvement_id,
            "workflow_canonical_improvement_label": canonical_intervention_label_map[target_canonical_improvement_id],
            "candidate_improvement_ids": list(target_frame.plan.improvement_candidate_ids),
            "candidate_improvement_labels": [
                improvement_label_map[improvement_id]
                for improvement_id in target_frame.plan.improvement_candidate_ids
            ],
            "candidate_canonical_improvement_ids": [
                _workflow_canonical_intervention_id(improvement_id)
                for improvement_id in target_frame.plan.improvement_candidate_ids
            ],
            "workflow_reward_score": round(target_frame.plan.primary_reward_score, 6),
            "workflow_reward_bucket_id": target_frame.plan.primary_reward_bucket_id,
            "workflow_reward_bucket_label": target_frame.plan.primary_reward_bucket_label,
            "workflow_reward_components": target_frame.plan.primary_reward_components,
            "trajectory_id": trajectory_id,
            "step_index": 0,
            "trace_length": 1,
        }
        target_action = {
            "name": "select_intervention_family",
            "action": {
                "intervention_family_id": target_canonical_improvement_id,
            },
        }
        text = _serialize_workflow_decision_step(
            task_id=trajectory_id,
            family="oscar_workflow_transfer",
            difficulty=5,
            trace_step="transfer_primary_intervention",
            step_index=0,
            trace_length=1,
            previous_action="<start>",
            candidate_bucket=(
                "family:oscar_workflow_transfer"
                f"|motif:{source_frame.plan.motif_id}"
                f"|kpi_family:{target_canonical_kpi_id}|step:0|prev:<start>"
            ),
            doc_title=target_frame.doc_title,
            doc_group=str(target_frame.record.metadata.get("doc_group", "unknown")),
            doc_id=target_frame.record.document_id,
            source_kind=str(target_frame.record.metadata.get("source_kind", "unknown")),
            source_view=target_frame.record.view,
            section_path=target_frame.section_path,
            question=(
                f"Which primary intervention best transfers from the source case to move KPI "
                f"{kpi_label_map[target_frame.plan.focus_kpi_id]} in the target case?"
            ),
            context_payload={
                "source_case": {
                    "environment": source_frame.profile.environment_label,
                    "excerpt": source_frame.body,
                    "motif": {
                        "id": source_frame.plan.motif_id,
                        "label": source_frame.plan.motif_label,
                    },
                    "focus_kpi": {
                        "id": source_frame.plan.focus_kpi_id,
                        "label": kpi_label_map[source_frame.plan.focus_kpi_id],
                        "canonical_id": source_canonical_kpi_id,
                        "canonical_label": canonical_kpi_label_map[source_canonical_kpi_id],
                    },
                    "primary_intervention": {
                        "id": source_frame.plan.primary_improvement_id,
                        "label": improvement_label_map[source_frame.plan.primary_improvement_id],
                        "canonical_id": source_canonical_improvement_id,
                        "canonical_label": canonical_intervention_label_map[source_canonical_improvement_id],
                    },
                    "reward_bucket": source_frame.plan.primary_reward_bucket_label,
                },
                "target_case": {
                    "environment": target_frame.profile.environment_label,
                    "excerpt": target_frame.body,
                    "focus_kpi": {
                        "id": target_frame.plan.focus_kpi_id,
                        "label": kpi_label_map[target_frame.plan.focus_kpi_id],
                        "canonical_id": target_canonical_kpi_id,
                        "canonical_label": canonical_kpi_label_map[target_canonical_kpi_id],
                    },
                    "active_bottlenecks": list(target_frame.plan.active_bottleneck_ids),
                    "candidate_improvements": [
                        {
                            "id": improvement_id,
                            "label": improvement_label_map[improvement_id],
                            "canonical_id": _workflow_canonical_intervention_id(improvement_id),
                            "canonical_label": canonical_intervention_label_map[_workflow_canonical_intervention_id(improvement_id)],
                        }
                        for improvement_id in target_frame.plan.improvement_candidate_ids
                    ],
                },
            },
            verifier_context={
                "workflow_motif_id": target_frame.plan.motif_id,
                "workflow_motif_label": motif_label_map[target_frame.plan.motif_id],
                "projected_reward_score": round(target_frame.plan.primary_reward_score, 6),
                "projected_reward_bucket": target_frame.plan.primary_reward_bucket_label,
                "should_stop": True,
            },
            auxiliary_targets=auxiliary_targets,
            target_action=target_action,
            task_kind="workflow_transfer",
            state_tokens=(
                *_state_tokens(
                    doc_group=str(target_frame.record.metadata.get("doc_group", "unknown")),
                    source_kind=str(target_frame.record.metadata.get("source_kind", "unknown")),
                    source_view=target_frame.record.view,
                    family="oscar_workflow_transfer",
                    section_depth=_section_depth(target_frame.section_path or target_frame.doc_title),
                    task_kind="workflow_transfer",
                ),
                f"workflow.motif={target_frame.plan.motif_id}",
                f"source.profile={source_frame.profile.profile_id}",
                f"target.profile={target_frame.profile.profile_id}",
                f"target.kpi_family={target_canonical_kpi_id}",
                f"source.intervention_family={source_canonical_improvement_id}",
            ),
            state_scalars=(
                float(len(set(source_frame.plan.active_bottleneck_ids).intersection(target_frame.plan.active_bottleneck_ids))),
                source_frame.plan.primary_reward_score,
                target_frame.plan.primary_reward_score,
            ),
        )
        task_groups.setdefault(source_frame.plan.motif_id, []).append(
            OscarScopeReasoningTask(
                task_id=trajectory_id,
                benchmark=OSCAR_SCOPE_REASONING_BENCHMARK,
                family="oscar_workflow_transfer",
                trace_step="transfer_primary_intervention",
                text=text,
                metadata=auxiliary_targets,
            )
        )
    return _interleave_task_groups(tuple(task_groups.values()))


def _round_robin_limit(
    task_groups: Sequence[Sequence[OscarScopeReasoningTask]],
    *,
    max_examples: int | None,
) -> list[OscarScopeReasoningTask]:
    ordered_groups = [list(group) for group in task_groups if group]
    if max_examples is None:
        return [task for group in ordered_groups for task in group]
    if max_examples <= 0 or not ordered_groups:
        return []
    selected: list[OscarScopeReasoningTask] = []
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


def _interleave_task_groups(
    task_groups: Sequence[Sequence[OscarScopeReasoningTask]],
) -> list[OscarScopeReasoningTask]:
    ordered_groups = [list(group) for group in task_groups if group]
    if not ordered_groups:
        return []
    selected: list[OscarScopeReasoningTask] = []
    offsets = [0 for _ in ordered_groups]
    while True:
        progressed = False
        for group_index, group in enumerate(ordered_groups):
            offset = offsets[group_index]
            if offset >= len(group):
                continue
            selected.append(group[offset])
            offsets[group_index] += 1
            progressed = True
        if not progressed:
            break
    return selected


def _drop_incomplete_trajectories(
    tasks: Sequence[OscarScopeReasoningTask],
) -> list[OscarScopeReasoningTask]:
    expected_lengths: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for task in tasks:
        trajectory_id = str(task.metadata.get("trajectory_id", task.task_id))
        selected_counts[trajectory_id] = selected_counts.get(trajectory_id, 0) + 1
        trace_length = int(task.metadata.get("trace_length", 1) or 1)
        expected_lengths[trajectory_id] = max(expected_lengths.get(trajectory_id, 1), trace_length)
    return [
        task
        for task in tasks
        if selected_counts.get(str(task.metadata.get("trajectory_id", task.task_id)), 0)
        >= expected_lengths.get(str(task.metadata.get("trajectory_id", task.task_id)), 1)
    ]


def build_oscar_scope_reasoning_tasks(
    *,
    roots: Sequence[str | Path] = (),
    paths: Sequence[str | Path] = (),
    auto_discover: bool = True,
    max_documents: int | None = None,
    max_examples: int | None = None,
    views: Sequence[str] = ("native_chunk", "section_outline"),
    families: Sequence[str] = OSCAR_SCOPE_REASONING_FAMILIES,
) -> tuple[OscarScopeReasoningTask, ...]:
    unsupported = sorted(family for family in families if family not in OSCAR_SCOPE_REASONING_FAMILIES)
    if unsupported:
        raise ValueError(f"Unsupported Oscar reasoning families: {unsupported}")
    records = build_oscar_scope_records(
        roots=roots,
        paths=paths,
        auto_discover=auto_discover,
        max_documents=max_documents,
        max_chunks=None,
        views=views,
    )
    if any(family in OSCAR_WORKFLOW_REASONING_FAMILIES for family in families):
        records = _merge_records(
            records,
            _workflow_scope_records(
                roots=roots,
                paths=paths,
                auto_discover=auto_discover,
                views=views,
            ),
        )
    graph_index = _build_document_graph_index(records)
    task_groups: list[list[OscarScopeReasoningTask]] = []
    if "oscar_section_anchor" in families:
        task_groups.append(_native_section_anchor_tasks(records, graph_index=graph_index))
    if "oscar_outline_next_heading" in families:
        task_groups.append(_outline_next_heading_tasks(records, graph_index=graph_index))
    if "oscar_concept_tags" in families:
        task_groups.append(_concept_tag_tasks(records, graph_index=graph_index))
    if "oscar_workflow_environment" in families:
        task_groups.append(_workflow_environment_tasks(records, graph_index=graph_index))
    if "oscar_workflow_kpi_tags" in families:
        task_groups.append(
            _workflow_multi_tag_tasks(
                records,
                graph_index=graph_index,
                family="oscar_workflow_kpi_tags",
                task_kind="workflow_kpi_tags",
                question="Which KPI families matter most for this workflow excerpt?",
                entries=_WORKFLOW_KPI_RULES,
                default_attr_name="default_kpis",
                output_key="kpis",
            )
        )
    if "oscar_workflow_bottleneck_tags" in families:
        task_groups.append(
            _workflow_multi_tag_tasks(
                records,
                graph_index=graph_index,
                family="oscar_workflow_bottleneck_tags",
                task_kind="workflow_bottleneck_tags",
                question="Which bottleneck classes are most likely active in this workflow excerpt?",
                entries=_WORKFLOW_BOTTLENECK_RULES,
                default_attr_name="default_bottlenecks",
                output_key="bottlenecks",
            )
        )
    if "oscar_workflow_improvement_tags" in families:
        task_groups.append(
            _workflow_multi_tag_tasks(
                records,
                graph_index=graph_index,
                family="oscar_workflow_improvement_tags",
                task_kind="workflow_improvement_tags",
                question="Which improvement levers best fit this workflow excerpt?",
                entries=_WORKFLOW_IMPROVEMENT_RULES,
                default_attr_name="default_improvements",
                output_key="improvements",
            )
        )
    if "oscar_workflow_kpi_improvement" in families:
        task_groups.append(_workflow_kpi_improvement_tasks(records, graph_index=graph_index))
    if "oscar_workflow_intervention_trace" in families:
        task_groups.append(_workflow_intervention_trace_tasks(records, graph_index=graph_index))
    if "oscar_workflow_case_analogy" in families:
        task_groups.append(_workflow_case_analogy_tasks(records, graph_index=graph_index))
    if "oscar_workflow_transfer" in families:
        task_groups.append(_workflow_transfer_tasks(records, graph_index=graph_index))
    tasks = _round_robin_limit(task_groups, max_examples=max_examples)
    tasks = _drop_incomplete_trajectories(tasks)
    return tuple(tasks)
