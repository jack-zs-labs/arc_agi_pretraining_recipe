"""ARC trajectory sampler workspace."""

from typing import TYPE_CHECKING

from .stage1_latent_sampler import TaskLatent, sample_latent_rule

if TYPE_CHECKING:
    from .gsm8k_reasoning_parser import ParserFailure
    from .mixed_reasoning_dataset import ReasoningTextExample
    from .mmlu_parser import MMLUExample, MMLUParserFailure
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
    if name in {
        "ReasoningTextExample",
        "build_arc_reasoning_examples",
        "build_gsm8k_reasoning_examples",
        "build_mmlu_reasoning_examples",
        "split_examples",
        "texts_from_examples",
    }:
        from .mixed_reasoning_dataset import (
            ReasoningTextExample,
            build_arc_reasoning_examples,
            build_gsm8k_reasoning_examples,
            build_mmlu_reasoning_examples,
            split_examples,
            texts_from_examples,
        )

        exports = {
            "ReasoningTextExample": ReasoningTextExample,
            "build_arc_reasoning_examples": build_arc_reasoning_examples,
            "build_gsm8k_reasoning_examples": build_gsm8k_reasoning_examples,
            "build_mmlu_reasoning_examples": build_mmlu_reasoning_examples,
            "split_examples": split_examples,
            "texts_from_examples": texts_from_examples,
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
    "EpisodeSpec",
    "EntitySpec",
    "EncodedWorkspace",
    "ExecutedEpisode",
    "ExecutedExample",
    "ExportSummary",
    "GoalSpec",
    "MMLUExample",
    "MMLUParserFailure",
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
    "build_mmlu_examples",
    "build_mmlu_reasoning_examples",
    "build_mmlu_trajectories",
    "build_trajectories",
    "build_word_problem_trajectories",
    "encode_workspace",
    "execute_episode",
    "parse_mmlu_row",
    "parse_gsm8k_row",
    "sample_episode",
    "sample_latent_rule",
    "sample_word_problem_example",
    "serialize_reasoning_state_text",
    "serialize_workspace_text",
    "split_examples",
    "texts_from_examples",
    "verifier_targets",
    "write_trm_dataset",
]
