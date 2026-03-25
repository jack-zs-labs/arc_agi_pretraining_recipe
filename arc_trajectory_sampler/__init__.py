"""ARC trajectory sampler workspace."""

from typing import TYPE_CHECKING

from .stage1_latent_sampler import TaskLatent, sample_latent_rule

if TYPE_CHECKING:
    from .core_graph_extractor import CoreCodeGraph, CoreGraphEdge, CoreGraphNode
    from .core_loader import CORERow, CoreCodeLine, CoreNodeRef
    from .core_reasoning_adapter import CoreReasoningTask
    from .gsm8k_reasoning_parser import ParserFailure
    from .mixed_reasoning_dataset import ReasoningTextExample
    from .mmlu_parser import MMLUExample, MMLUParserFailure
    from .mmlu_variants import REDUX_LABEL_MODE_CHOICES
    from .olympiad_math_parser import OlympiadMathExample, OlympiadMathParserFailure
    from .reasoning_ir import AbstractReasoningTask, ChoiceSpec, EntitySpec, GoalSpec, QuantitySpec
    from .state_adapter import EncodedWorkspace
    from .stage2_episode_sampler import EpisodeSpec
    from .stage3_grid_executor import ExecutedEpisode, ExecutedExample
    from .stage4_trajectory_dataset import TrajectoryRecord
    from .trm_dataset_export import ExportSummary
    from .word_problem_translation_dataset import WordProblemExample


def __getattr__(name: str):
    if name in {"AbstractReasoningTask", "ChoiceSpec", "EntitySpec", "GoalSpec", "QuantitySpec"}:
        from .reasoning_ir import AbstractReasoningTask, ChoiceSpec, EntitySpec, GoalSpec, QuantitySpec

        exports = {
            "AbstractReasoningTask": AbstractReasoningTask,
            "ChoiceSpec": ChoiceSpec,
            "EntitySpec": EntitySpec,
            "GoalSpec": GoalSpec,
            "QuantitySpec": QuantitySpec,
        }
        return exports[name]
    if name in {"EpisodeSpec", "sample_episode"}:
        from .stage2_episode_sampler import EpisodeSpec, sample_episode

        exports = {
            "EpisodeSpec": EpisodeSpec,
            "sample_episode": sample_episode,
        }
        return exports[name]
    if name in {"ExecutedEpisode", "ExecutedExample", "execute_episode"}:
        from .stage3_grid_executor import ExecutedEpisode, ExecutedExample, execute_episode

        exports = {
            "ExecutedEpisode": ExecutedEpisode,
            "ExecutedExample": ExecutedExample,
            "execute_episode": execute_episode,
        }
        return exports[name]
    if name in {"TrajectoryRecord", "build_trajectories"}:
        from .stage4_trajectory_dataset import TrajectoryRecord, build_trajectories

        exports = {
            "TrajectoryRecord": TrajectoryRecord,
            "build_trajectories": build_trajectories,
        }
        return exports[name]
    if name in {
        "EncodedWorkspace",
        "encode_workspace",
        "serialize_reasoning_state_text",
        "serialize_workspace_text",
        "verifier_targets",
    }:
        from .state_adapter import (
            EncodedWorkspace,
            encode_workspace,
            serialize_reasoning_state_text,
            serialize_workspace_text,
            verifier_targets,
        )

        exports = {
            "EncodedWorkspace": EncodedWorkspace,
            "encode_workspace": encode_workspace,
            "serialize_reasoning_state_text": serialize_reasoning_state_text,
            "serialize_workspace_text": serialize_workspace_text,
            "verifier_targets": verifier_targets,
        }
        return exports[name]
    if name in {"ExportSummary", "write_trm_dataset"}:
        from .trm_dataset_export import ExportSummary, write_trm_dataset

        exports = {
            "ExportSummary": ExportSummary,
            "write_trm_dataset": write_trm_dataset,
        }
        return exports[name]
    if name in {"WordProblemExample", "build_word_problem_trajectories", "sample_word_problem_example"}:
        from .word_problem_translation_dataset import (
            WordProblemExample,
            build_word_problem_trajectories,
            sample_word_problem_example,
        )

        exports = {
            "WordProblemExample": WordProblemExample,
            "build_word_problem_trajectories": build_word_problem_trajectories,
            "sample_word_problem_example": sample_word_problem_example,
        }
        return exports[name]
    if name in {"ParserFailure", "build_gsm8k_examples", "build_gsm8k_trajectories", "parse_gsm8k_row"}:
        from .gsm8k_reasoning_parser import (
            ParserFailure,
            build_gsm8k_examples,
            build_gsm8k_trajectories,
            parse_gsm8k_row,
        )

        exports = {
            "ParserFailure": ParserFailure,
            "build_gsm8k_examples": build_gsm8k_examples,
            "build_gsm8k_trajectories": build_gsm8k_trajectories,
            "parse_gsm8k_row": parse_gsm8k_row,
        }
        return exports[name]
    if name in {"CoreCodeGraph", "CoreGraphEdge", "CoreGraphNode", "extract_core_code_graph"}:
        from .core_graph_extractor import CoreCodeGraph, CoreGraphEdge, CoreGraphNode, extract_core_code_graph

        exports = {
            "CoreCodeGraph": CoreCodeGraph,
            "CoreGraphEdge": CoreGraphEdge,
            "CoreGraphNode": CoreGraphNode,
            "extract_core_code_graph": extract_core_code_graph,
        }
        return exports[name]
    if name in {
        "CORERow",
        "CoreCodeLine",
        "CoreNodeRef",
        "iter_core_supervision_texts",
        "load_core_rows",
        "serialize_core_supervision_text",
    }:
        from .core_loader import (
            CORERow,
            CoreCodeLine,
            CoreNodeRef,
            iter_core_supervision_texts,
            load_core_rows,
            serialize_core_supervision_text,
        )

        exports = {
            "CORERow": CORERow,
            "CoreCodeLine": CoreCodeLine,
            "CoreNodeRef": CoreNodeRef,
            "iter_core_supervision_texts": iter_core_supervision_texts,
            "load_core_rows": load_core_rows,
            "serialize_core_supervision_text": serialize_core_supervision_text,
        }
        return exports[name]
    if name in {"CoreReasoningTask", "build_core_reasoning_tasks", "serialize_core_reasoning_task"}:
        from .core_reasoning_adapter import CoreReasoningTask, build_core_reasoning_tasks, serialize_core_reasoning_task

        exports = {
            "CoreReasoningTask": CoreReasoningTask,
            "build_core_reasoning_tasks": build_core_reasoning_tasks,
            "serialize_core_reasoning_task": serialize_core_reasoning_task,
        }
        return exports[name]
    if name in {
        "ReasoningTextExample",
        "build_arc_reasoning_examples",
        "build_core_reasoning_examples",
        "build_gsm8k_reasoning_examples",
        "build_mmlu_reasoning_examples",
        "build_mmlu_pro_reasoning_examples",
        "build_mmlu_redux_reasoning_examples",
        "build_oscar_graph_reasoning_examples",
        "build_oscar_scope_examples",
        "build_oscar_scope_reasoning_examples",
        "build_olympiad_math_pretraining_examples",
        "build_olympiad_math_reasoning_examples",
        "build_ptraj_examples",
        "split_examples",
        "texts_from_examples",
    }:
        from .mixed_reasoning_dataset import (
            ReasoningTextExample,
            build_arc_reasoning_examples,
            build_core_reasoning_examples,
            build_gsm8k_reasoning_examples,
            build_mmlu_reasoning_examples,
            build_mmlu_pro_reasoning_examples,
            build_mmlu_redux_reasoning_examples,
            build_oscar_graph_reasoning_examples,
            build_oscar_scope_examples,
            build_oscar_scope_reasoning_examples,
            build_olympiad_math_pretraining_examples,
            build_olympiad_math_reasoning_examples,
            build_ptraj_examples,
            split_examples,
            texts_from_examples,
        )

        exports = {
            "ReasoningTextExample": ReasoningTextExample,
            "build_arc_reasoning_examples": build_arc_reasoning_examples,
            "build_core_reasoning_examples": build_core_reasoning_examples,
            "build_gsm8k_reasoning_examples": build_gsm8k_reasoning_examples,
            "build_mmlu_reasoning_examples": build_mmlu_reasoning_examples,
            "build_mmlu_pro_reasoning_examples": build_mmlu_pro_reasoning_examples,
            "build_mmlu_redux_reasoning_examples": build_mmlu_redux_reasoning_examples,
            "build_oscar_graph_reasoning_examples": build_oscar_graph_reasoning_examples,
            "build_oscar_scope_examples": build_oscar_scope_examples,
            "build_oscar_scope_reasoning_examples": build_oscar_scope_reasoning_examples,
            "build_olympiad_math_pretraining_examples": build_olympiad_math_pretraining_examples,
            "build_olympiad_math_reasoning_examples": build_olympiad_math_reasoning_examples,
            "build_ptraj_examples": build_ptraj_examples,
            "split_examples": split_examples,
            "texts_from_examples": texts_from_examples,
        }
        return exports[name]
    if name in {
        "OlympiadMathExample",
        "OlympiadMathParserFailure",
        "build_olympiad_math_examples",
        "build_olympiad_math_trajectories",
        "parse_olympiad_math_row",
    }:
        from .olympiad_math_parser import (
            OlympiadMathExample,
            OlympiadMathParserFailure,
            build_olympiad_math_examples,
            build_olympiad_math_trajectories,
            parse_olympiad_math_row,
        )

        exports = {
            "OlympiadMathExample": OlympiadMathExample,
            "OlympiadMathParserFailure": OlympiadMathParserFailure,
            "build_olympiad_math_examples": build_olympiad_math_examples,
            "build_olympiad_math_trajectories": build_olympiad_math_trajectories,
            "parse_olympiad_math_row": parse_olympiad_math_row,
        }
        return exports[name]
    if name in {
        "build_mmlu_pro_examples",
        "build_mmlu_pro_trajectories",
        "build_mmlu_redux_examples",
        "build_mmlu_redux_trajectories",
        "REDUX_LABEL_MODE_CHOICES",
    }:
        from .mmlu_variants import (
            REDUX_LABEL_MODE_CHOICES,
            build_mmlu_pro_examples,
            build_mmlu_pro_trajectories,
            build_mmlu_redux_examples,
            build_mmlu_redux_trajectories,
        )

        exports = {
            "build_mmlu_pro_examples": build_mmlu_pro_examples,
            "build_mmlu_pro_trajectories": build_mmlu_pro_trajectories,
            "build_mmlu_redux_examples": build_mmlu_redux_examples,
            "build_mmlu_redux_trajectories": build_mmlu_redux_trajectories,
            "REDUX_LABEL_MODE_CHOICES": REDUX_LABEL_MODE_CHOICES,
        }
        return exports[name]
    if name in {"MMLUExample", "MMLUParserFailure", "build_mmlu_examples", "build_mmlu_trajectories", "parse_mmlu_row"}:
        from .mmlu_parser import (
            MMLUExample,
            MMLUParserFailure,
            build_mmlu_examples,
            build_mmlu_trajectories,
            parse_mmlu_row,
        )

        exports = {
            "MMLUExample": MMLUExample,
            "MMLUParserFailure": MMLUParserFailure,
            "build_mmlu_examples": build_mmlu_examples,
            "build_mmlu_trajectories": build_mmlu_trajectories,
            "parse_mmlu_row": parse_mmlu_row,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractReasoningTask",
    "ChoiceSpec",
    "CoreCodeGraph",
    "CORERow",
    "CoreCodeLine",
    "CoreGraphEdge",
    "CoreGraphNode",
    "CoreNodeRef",
    "CoreReasoningTask",
    "EpisodeSpec",
    "EntitySpec",
    "EncodedWorkspace",
    "ExecutedEpisode",
    "ExecutedExample",
    "ExportSummary",
    "GoalSpec",
    "MMLUExample",
    "MMLUParserFailure",
    "OlympiadMathExample",
    "OlympiadMathParserFailure",
    "ParserFailure",
    "QuantitySpec",
    "ReasoningTextExample",
    "TaskLatent",
    "TrajectoryRecord",
    "WordProblemExample",
    "build_gsm8k_examples",
    "build_gsm8k_reasoning_examples",
    "build_gsm8k_trajectories",
    "build_arc_reasoning_examples",
    "build_core_reasoning_tasks",
    "build_core_reasoning_examples",
    "build_mmlu_examples",
    "build_mmlu_pro_examples",
    "build_mmlu_pro_reasoning_examples",
    "build_mmlu_pro_trajectories",
    "build_mmlu_reasoning_examples",
    "build_mmlu_redux_examples",
    "build_mmlu_redux_reasoning_examples",
    "build_mmlu_redux_trajectories",
    "build_mmlu_trajectories",
    "build_oscar_graph_reasoning_examples",
    "build_oscar_scope_examples",
    "build_oscar_scope_reasoning_examples",
    "build_olympiad_math_examples",
    "build_olympiad_math_pretraining_examples",
    "build_olympiad_math_reasoning_examples",
    "build_olympiad_math_trajectories",
    "build_ptraj_examples",
    "build_trajectories",
    "build_word_problem_trajectories",
    "encode_workspace",
    "extract_core_code_graph",
    "execute_episode",
    "iter_core_supervision_texts",
    "load_core_rows",
    "parse_mmlu_row",
    "parse_olympiad_math_row",
    "parse_gsm8k_row",
    "sample_episode",
    "sample_latent_rule",
    "sample_word_problem_example",
    "serialize_core_reasoning_task",
    "serialize_core_supervision_text",
    "serialize_reasoning_state_text",
    "serialize_workspace_text",
    "REDUX_LABEL_MODE_CHOICES",
    "split_examples",
    "texts_from_examples",
    "verifier_targets",
    "write_trm_dataset",
]
