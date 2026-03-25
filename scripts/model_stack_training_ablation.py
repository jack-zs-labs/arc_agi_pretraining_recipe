from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterable

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This training ablation script requires torch. Install requirements-models.txt or use .venv_atari.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler import build_trajectories, sample_episode, sample_latent_rule
from arc_trajectory_sampler.state_adapter import encode_workspace, serialize_workspace_text, verifier_targets
from models import AttentionBackendConfig, DecoderLanguageModel, DecoderModelConfig, InferenceBudget
from models.reasoning_tokenizer import (
    ReasoningTokenizer,
    add_tokenizer_cli_arguments,
    build_reasoning_tokenizer,
)

ARC_SERIALIZATION_CHOICES = ("compact_json", "structured_workspace")
TEXT_TOKENIZER: ReasoningTokenizer | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny token-level training comparison for SDPA, hybrid, and MLA.")
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--val-episodes", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--sliding-window", type=int, default=64)
    parser.add_argument("--global-token-stride", type=int, default=8)
    parser.add_argument("--sink-token-count", type=int, default=2)
    parser.add_argument("--latent-kv-dim", type=int, default=48)
    parser.add_argument("--mla-latent-kv-dims", nargs="+", type=int, default=None)
    parser.add_argument("--scale-invariant-tau", type=float, default=10.0)
    parser.add_argument(
        "--scale-invariant-last-n-layers",
        nargs="+",
        type=int,
        default=None,
        help="Optional sweep of suffix-layer counts for SIA backends. Omit to use the full decoder depth.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--arc-serialization",
        choices=ARC_SERIALIZATION_CHOICES,
        default="compact_json",
        help="ARC text view for the decoder ablation. The default preserves the old compact trajectory JSON.",
    )
    parser.add_argument(
        "--arc-include-verifier-targets",
        action="store_true",
        help="Append verifier supervision to structured ARC text records.",
    )
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--csv-output", type=str, default="", help="Optional CSV output path.")
    add_tokenizer_cli_arguments(parser, default_kind="epiplex", default_vocab_size=4096, default_task="generic")
    return parser.parse_args()


def auto_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def set_text_tokenizer(tokenizer: ReasoningTokenizer) -> None:
    global TEXT_TOKENIZER
    TEXT_TOKENIZER = tokenizer


def require_text_tokenizer() -> ReasoningTokenizer:
    if TEXT_TOKENIZER is None:
        raise RuntimeError("Text tokenizer has not been initialized yet.")
    return TEXT_TOKENIZER


def seeds_for_run(args: argparse.Namespace) -> list[int]:
    return args.seeds if args.seeds is not None else [args.seed]


def ordered_backends() -> list[str]:
    return ["sdpa", "hybrid", "sia", "sia_hybrid", "mla", "mla_sia"]


def uses_hybrid_mask(backend: str) -> bool:
    return backend in {"hybrid", "sia_hybrid"}


def uses_scale_invariant_scores(backend: str) -> bool:
    return backend in {"sia", "sia_hybrid", "mla_sia"}


def uses_latent_kv(backend: str) -> bool:
    return backend in {"mla", "mla_sia"}


def scale_invariant_last_n_layers_for_run(args: argparse.Namespace) -> list[int | None]:
    return args.scale_invariant_last_n_layers if args.scale_invariant_last_n_layers is not None else [None]


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_runs(results: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in results["runs"]:
        rows.append(
            {
                "device": results["device"],
                "seed": run["seed"],
                "backend": run["backend"],
                "arc_serialization": results["arc_serialization"],
                "arc_include_verifier_targets": results["arc_include_verifier_targets"],
                "tokenizer_kind": results.get("tokenizer_kind"),
                "tokenizer_vocab_size": results.get("tokenizer_vocab_size"),
                "seq_len": run["seq_len"],
                "latent_kv_dim": run["latent_kv_dim"] if run["latent_kv_dim"] is not None else "",
                "scale_invariant_last_n_layers": (
                    run["scale_invariant_last_n_layers"] if run["scale_invariant_last_n_layers"] is not None else ""
                ),
                "parameter_count": run["parameter_count"],
                "initial_val_loss": run["initial_val_loss"],
                "final_val_loss": run["final_val_loss"],
                "train_tokens_per_second": run["train_tokens_per_second"],
                "train_elapsed_seconds": run["train_elapsed_seconds"],
                "cache_numel_for_probe": run["cache_numel_for_probe"],
            }
        )
    return rows


def serialize_action_target(name: str, action: dict[str, object]) -> str:
    return json.dumps(
        {
            "name": name,
            "action": action,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def sample_arc_texts(
    *,
    num_episodes: int,
    seed_start: int,
    serialization: str,
    include_verifier_targets: bool,
) -> list[str]:
    texts: list[str] = []
    for offset in range(num_episodes):
        seed = seed_start + offset
        latent = sample_latent_rule(seed=seed)
        episode = sample_episode(latent, seed=seed)
        for record in build_trajectories(episode, include_test=True):
            if serialization == "compact_json":
                payload = record.to_jsonable()
                compact = {
                    "family": payload["family"],
                    "difficulty": payload["difficulty"],
                    "trace_template": payload["trace_template"],
                    "input_grid": payload["input_grid"],
                    "output_grid": payload["output_grid"],
                }
                texts.append("ARC\n" + json.dumps(compact, separators=(",", ":"), sort_keys=True) + "\n")
                continue
            previous_action = "<start>"
            for step_index, step in enumerate(record.steps):
                encoded = encode_workspace(record, step_index, include_verifier=False)
                target_action = serialize_action_target(step.name, step.action)
                texts.append(
                    serialize_workspace_text(
                        encoded,
                        previous_action=previous_action,
                        target_action=target_action,
                        verifier_state=verifier_targets(record, step_index) if include_verifier_targets else None,
                    )
                )
                previous_action = target_action
    return texts


def encode_text(text: str) -> list[int]:
    return require_text_tokenizer().encode(text, add_eos=True)


def chunk_token_stream(texts: Iterable[str], *, seq_len: int) -> list[list[int]]:
    tokenizer = require_text_tokenizer()
    windows: list[list[int]] = []
    stride = max(seq_len // 2, 1)
    for text in texts:
        tokens = encode_text(text)
        if len(tokens) < 2:
            continue
        if len(tokens) <= seq_len + 1:
            padded = tokens + [tokenizer.window_pad_token_id] * max(0, seq_len + 1 - len(tokens))
            windows.append(padded[: seq_len + 1])
            continue
        for start in range(0, len(tokens) - 1, stride):
            window = tokens[start : start + seq_len + 1]
            if len(window) < seq_len + 1:
                window = window + [tokenizer.window_pad_token_id] * (seq_len + 1 - len(window))
            windows.append(window[: seq_len + 1])
            if start + seq_len + 1 >= len(tokens):
                break
    return windows


def make_batches(windows: list[list[int]], *, batch_size: int, rng: random.Random) -> list[list[list[int]]]:
    shuffled = windows[:]
    rng.shuffle(shuffled)
    return [shuffled[index : index + batch_size] for index in range(0, len(shuffled), batch_size)]


def build_model_config(args: argparse.Namespace, *, backend: str, vocab_size: int) -> DecoderModelConfig:
    return build_model_config_for_run(
        args,
        backend=backend,
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        latent_kv_dim=args.latent_kv_dim if uses_latent_kv(backend) else None,
        scale_invariant_last_n_layers=None,
    )


def build_model_config_for_run(
    args: argparse.Namespace,
    *,
    backend: str,
    vocab_size: int,
    seq_len: int,
    latent_kv_dim: int | None,
    scale_invariant_last_n_layers: int | None,
) -> DecoderModelConfig:
    if backend == "mla":
        attention = AttentionBackendConfig.from_preset(
            "mla_default",
            latent_kv_dim=latent_kv_dim,
        )
    elif backend == "mla_sia" and scale_invariant_last_n_layers == 1:
        attention = AttentionBackendConfig.from_preset(
            "mla_sia_prefill_l1",
            latent_kv_dim=latent_kv_dim,
            scale_invariant_tau=args.scale_invariant_tau,
        )
    else:
        attention = AttentionBackendConfig(
            backend=backend,
            dropout=0.0,
            sliding_window=args.sliding_window if uses_hybrid_mask(backend) else None,
            global_token_stride=args.global_token_stride if uses_hybrid_mask(backend) else None,
            sink_token_count=args.sink_token_count if uses_hybrid_mask(backend) else 0,
            latent_kv_dim=latent_kv_dim if uses_latent_kv(backend) else None,
            scale_invariant_tau=args.scale_invariant_tau if uses_scale_invariant_scores(backend) else None,
            scale_invariant_last_n_layers=scale_invariant_last_n_layers if uses_scale_invariant_scores(backend) else None,
        )
    return DecoderModelConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max(seq_len + 32, 512),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        attention=attention,
    )


def evaluate_loss(
    model: DecoderLanguageModel,
    windows: list[list[int]],
    *,
    batch_size: int,
    device: torch.device,
    attention_window: int | None,
) -> float:
    if not windows:
        return float("nan")
    model.eval()
    losses: list[float] = []
    budget = InferenceBudget(use_kv_cache=False, attention_window=attention_window)
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)
            inputs = batch_tensor[:, :-1]
            labels = batch_tensor[:, 1:]
            logits = model.forward(inputs, budget=budget).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def run_backend(
    *,
    backend: str,
    args: argparse.Namespace,
    seed: int,
    vocab_size: int,
    seq_len: int,
    latent_kv_dim: int | None,
    scale_invariant_last_n_layers: int | None,
    train_windows: list[list[int]],
    val_windows: list[list[int]],
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = DecoderLanguageModel(
        build_model_config_for_run(
            args,
            backend=backend,
            vocab_size=vocab_size,
            seq_len=seq_len,
            latent_kv_dim=latent_kv_dim,
            scale_invariant_last_n_layers=scale_invariant_last_n_layers,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    attention_window = args.sliding_window if uses_hybrid_mask(backend) else None
    budget = InferenceBudget(use_kv_cache=False, attention_window=attention_window)
    rng = random.Random(seed)
    batches = make_batches(train_windows, batch_size=args.batch_size, rng=rng)
    if not batches:
        raise ValueError("Training windows are empty.")

    initial_val_loss = evaluate_loss(
        model,
        val_windows,
        batch_size=args.batch_size,
        device=device,
        attention_window=attention_window,
    )
    step_losses: list[float] = []
    tokens_processed = 0
    model.train()
    synchronize_device(device)
    start_time = time.perf_counter()
    for step in range(args.steps):
        batch = batches[step % len(batches)]
        batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)
        inputs = batch_tensor[:, :-1]
        labels = batch_tensor[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward(inputs, budget=budget).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        step_losses.append(float(loss.item()))
        tokens_processed += int(labels.numel())
    synchronize_device(device)
    elapsed = time.perf_counter() - start_time
    final_val_loss = evaluate_loss(
        model,
        val_windows,
        batch_size=args.batch_size,
        device=device,
        attention_window=attention_window,
    )

    cache_probe = torch.tensor(val_windows[0][:seq_len], dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        cache_output = model.forward(
            cache_probe[:, :seq_len],
            budget=InferenceBudget(
                use_kv_cache=True,
                attention_window=attention_window,
                max_cache_tokens=seq_len,
            ),
        )
    cache_numel = 0
    if cache_output.cache is not None:
        for layer in cache_output.cache.layers:
            for name in ("key", "value", "latent"):
                tensor = getattr(layer, name, None)
                if tensor is not None:
                    cache_numel += int(tensor.numel())

    return {
        "seed": seed,
        "backend": backend,
        "seq_len": seq_len,
        "latent_kv_dim": latent_kv_dim,
        "scale_invariant_last_n_layers": scale_invariant_last_n_layers,
        "parameter_count": parameter_count(model),
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "train_loss_history": step_losses,
        "train_tokens_per_second": float(tokens_processed / max(elapsed, 1e-9)),
        "train_elapsed_seconds": elapsed,
        "cache_numel_for_probe": cache_numel,
    }


def main() -> None:
    args = parse_args()
    device = auto_device()
    seq_lens = args.seq_lens if args.seq_lens is not None else [args.seq_len]
    mla_latent_kv_dims = args.mla_latent_kv_dims if args.mla_latent_kv_dims is not None else [args.latent_kv_dim]
    scale_invariant_last_n_layers = scale_invariant_last_n_layers_for_run(args)
    seeds = seeds_for_run(args)
    dataset_runs: list[dict[str, object]] = []
    sweep_results: list[dict[str, object]] = []
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        train_texts = sample_arc_texts(
            num_episodes=args.train_episodes,
            seed_start=seed,
            serialization=args.arc_serialization,
            include_verifier_targets=args.arc_include_verifier_targets,
        )
        val_texts = sample_arc_texts(
            num_episodes=args.val_episodes,
            seed_start=seed + 10_000,
            serialization=args.arc_serialization,
            include_verifier_targets=args.arc_include_verifier_targets,
        )
        tokenizer = build_reasoning_tokenizer(
            train_texts,
            kind=args.tokenizer,
            vocab_size=args.tokenizer_vocab_size,
            task=args.tokenizer_task,
            min_freq=args.tokenizer_min_freq,
            candidate_pool_size=args.tokenizer_candidate_pool_size,
            max_piece_chars=args.tokenizer_max_piece_chars,
            fit_workers=args.tokenizer_fit_workers,
            fit_verbose=args.tokenizer_fit_verbose,
            load_path=args.tokenizer_load,
            save_path=args.tokenizer_save,
        )
        set_text_tokenizer(tokenizer)
        windows_by_seq_len = {
            seq_len: {
                "train": chunk_token_stream(train_texts, seq_len=seq_len),
                "val": chunk_token_stream(val_texts, seq_len=seq_len),
            }
            for seq_len in seq_lens
        }
        dataset_runs.append(
            {
                "seed": seed,
                "arc_serialization": args.arc_serialization,
                "arc_include_verifier_targets": args.arc_include_verifier_targets,
                "train_examples": len(train_texts),
                "val_examples": len(val_texts),
                "tokenizer": tokenizer.summary(),
                "windows_by_seq_len": {
                    str(seq_len): {
                        "train": len(data["train"]),
                        "val": len(data["val"]),
                    }
                    for seq_len, data in windows_by_seq_len.items()
                },
            }
        )
        for seq_len in seq_lens:
            train_windows = windows_by_seq_len[seq_len]["train"]
            val_windows = windows_by_seq_len[seq_len]["val"]
            for backend in ordered_backends():
                scale_invariant_options = scale_invariant_last_n_layers if uses_scale_invariant_scores(backend) else [None]
                for layer_count in scale_invariant_options:
                    if uses_latent_kv(backend):
                        for latent_kv_dim in mla_latent_kv_dims:
                            sweep_results.append(
                                run_backend(
                                    backend=backend,
                                    args=args,
                                    seed=seed,
                                    vocab_size=tokenizer.vocab_size,
                                    seq_len=seq_len,
                                    latent_kv_dim=latent_kv_dim,
                                    scale_invariant_last_n_layers=layer_count,
                                    train_windows=train_windows,
                                    val_windows=val_windows,
                                    device=device,
                                )
                            )
                        continue
                    sweep_results.append(
                        run_backend(
                            backend=backend,
                            args=args,
                            seed=seed,
                            vocab_size=tokenizer.vocab_size,
                            seq_len=seq_len,
                            latent_kv_dim=None,
                            scale_invariant_last_n_layers=layer_count,
                            train_windows=train_windows,
                            val_windows=val_windows,
                            device=device,
                        )
                    )
    results = {
        "device": device.type,
        "seeds": seeds,
        "seq_lens": seq_lens,
        "mla_latent_kv_dims": mla_latent_kv_dims,
        "scale_invariant_last_n_layers": scale_invariant_last_n_layers,
        "scale_invariant_tau": args.scale_invariant_tau,
        "arc_serialization": args.arc_serialization,
        "arc_include_verifier_targets": args.arc_include_verifier_targets,
        "tokenizer_kind": args.tokenizer,
        "tokenizer_vocab_size": (
            dataset_runs[0]["tokenizer"]["vocab_size"]
            if dataset_runs
            else args.tokenizer_vocab_size
        ),
        "datasets": dataset_runs,
        "runs": sweep_results,
        "backends": sweep_results,
    }
    payload = json.dumps(results, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    if args.csv_output:
        write_csv(args.csv_output, flatten_runs(results))
    print(payload)


if __name__ == "__main__":
    main()
