from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This ablation script requires torch. Install requirements-models.txt or use .venv_atari.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import AttentionBackendConfig, DecoderLanguageModel, DecoderModelConfig, InferenceBudget
from models.cache import DecoderKVCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SDPA, hybrid, and MLA reference backends.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--sliding-window", type=int, default=32)
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
    parser.add_argument(
        "--scale-invariant-decode-modes",
        nargs="+",
        choices=("all_tokens", "prefill_only"),
        default=None,
        help="Optional sweep of decode-time score modes for SIA backends.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timing-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--csv-output", type=str, default="", help="Optional CSV output path.")
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


def latency_summary(samples_ms: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def cache_stats(cache: object) -> dict[str, object]:
    if cache is None:
        return {
            "tokens_seen": 0,
            "current_length": 0,
            "bytes_total": 0,
            "numel_total": 0,
            "layer_breakdown": [],
        }
    layer_breakdown: list[dict[str, int]] = []
    bytes_total = 0
    numel_total = 0
    for index, layer in enumerate(cache.layers):
        layer_bytes = 0
        layer_numel = 0
        item: dict[str, int] = {"layer": index}
        for name in ("key", "value", "latent", "positions"):
            tensor = getattr(layer, name, None)
            nbytes = tensor_nbytes(tensor)
            numel = int(tensor.numel()) if tensor is not None else 0
            item[f"{name}_bytes"] = nbytes
            item[f"{name}_numel"] = numel
            layer_bytes += nbytes
            layer_numel += numel
        item["bytes_total"] = layer_bytes
        item["numel_total"] = layer_numel
        layer_breakdown.append(item)
        bytes_total += layer_bytes
        numel_total += layer_numel
    return {
        "tokens_seen": cache.tokens_seen,
        "current_length": cache.current_length,
        "bytes_total": bytes_total,
        "numel_total": numel_total,
        "layer_breakdown": layer_breakdown,
    }


def prompt_lengths_for_run(args: argparse.Namespace) -> list[int]:
    return args.prompt_lengths if args.prompt_lengths is not None else [args.prompt_length]


def mla_latent_kv_dims_for_run(args: argparse.Namespace) -> list[int]:
    return args.mla_latent_kv_dims if args.mla_latent_kv_dims is not None else [args.latent_kv_dim]


def scale_invariant_last_n_layers_for_run(args: argparse.Namespace) -> list[int | None]:
    return args.scale_invariant_last_n_layers if args.scale_invariant_last_n_layers is not None else [None]


def scale_invariant_decode_modes_for_run(args: argparse.Namespace) -> list[str]:
    return args.scale_invariant_decode_modes if args.scale_invariant_decode_modes is not None else ["all_tokens"]


def ordered_backends() -> list[str]:
    return ["sdpa", "hybrid", "sia", "sia_hybrid", "mla", "mla_sia"]


def uses_hybrid_mask(backend: str) -> bool:
    return backend in {"hybrid", "sia_hybrid"}


def uses_scale_invariant_scores(backend: str) -> bool:
    return backend in {"sia", "sia_hybrid", "mla_sia"}


def uses_latent_kv(backend: str) -> bool:
    return backend in {"mla", "mla_sia"}


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
        cache = run["cache"]
        forward_latency = run["forward_latency"]
        generate_latency = run["generate_latency"]
        rows.append(
            {
                "device": results["device"],
                "backend": run["backend"],
                "prompt_length": run["prompt_length"],
                "latent_kv_dim": run["latent_kv_dim"] if run["latent_kv_dim"] is not None else "",
                "scale_invariant_last_n_layers": (
                    run["scale_invariant_last_n_layers"] if run["scale_invariant_last_n_layers"] is not None else ""
                ),
                "scale_invariant_decode_mode": (
                    run["scale_invariant_decode_mode"] if run["scale_invariant_decode_mode"] is not None else ""
                ),
                "cache_tokens_seen": cache["tokens_seen"],
                "cache_current_length": cache["current_length"],
                "cache_bytes_total": cache["bytes_total"],
                "cache_numel_total": cache["numel_total"],
                "forward_latency_median_ms": forward_latency["median_ms"],
                "forward_latency_mean_ms": forward_latency["mean_ms"],
                "forward_latency_min_ms": forward_latency["min_ms"],
                "forward_latency_max_ms": forward_latency["max_ms"],
                "generate_latency_median_ms": generate_latency["median_ms"],
                "generate_latency_mean_ms": generate_latency["mean_ms"],
                "generate_latency_min_ms": generate_latency["min_ms"],
                "generate_latency_max_ms": generate_latency["max_ms"],
                "cached_decode_latency_median_ms": run["cached_decode_latency"]["median_ms"],
                "cached_decode_latency_mean_ms": run["cached_decode_latency"]["mean_ms"],
                "cached_decode_latency_min_ms": run["cached_decode_latency"]["min_ms"],
                "cached_decode_latency_max_ms": run["cached_decode_latency"]["max_ms"],
                "forward_tokens_per_second_median": run["forward_tokens_per_second_median"],
                "generate_new_tokens_per_second_median": run["generate_new_tokens_per_second_median"],
                "cached_decode_new_tokens_per_second_median": run["cached_decode_new_tokens_per_second_median"],
            }
        )
    return rows


def build_model_config(
    args: argparse.Namespace,
    *,
    backend: str,
    prompt_length: int,
    latent_kv_dim: int | None,
    scale_invariant_last_n_layers: int | None,
    scale_invariant_decode_mode: str,
) -> DecoderModelConfig:
    if backend == "mla":
        attention = AttentionBackendConfig.from_preset(
            "mla_default",
            latent_kv_dim=latent_kv_dim,
        )
    elif (
        backend == "mla_sia"
        and scale_invariant_last_n_layers == 1
        and scale_invariant_decode_mode == "prefill_only"
    ):
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
            scale_invariant_decode_mode=scale_invariant_decode_mode if uses_scale_invariant_scores(backend) else "all_tokens",
        )
    return DecoderModelConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=max(512, prompt_length + args.max_new_tokens + 32),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        attention=attention,
    )


def run_backend(
    *,
    backend: str,
    args: argparse.Namespace,
    prompt: torch.Tensor,
    device: torch.device,
    prompt_length: int,
    latent_kv_dim: int | None,
    scale_invariant_last_n_layers: int | None,
    scale_invariant_decode_mode: str,
) -> dict[str, object]:
    model = DecoderLanguageModel(
        build_model_config(
            args,
            backend=backend,
            prompt_length=prompt_length,
            latent_kv_dim=latent_kv_dim,
            scale_invariant_last_n_layers=scale_invariant_last_n_layers,
            scale_invariant_decode_mode=scale_invariant_decode_mode,
        )
    ).to(device)
    model.eval()
    budget = InferenceBudget(
        max_new_tokens=args.max_new_tokens,
        max_cache_tokens=prompt_length + args.max_new_tokens,
        attention_window=args.sliding_window if uses_hybrid_mask(backend) else None,
        use_kv_cache=True,
    )
    with torch.inference_mode():
        outputs = model.forward(prompt, budget=budget)
        generated = model.generate(prompt[:1], budget=budget)
        for _ in range(args.warmup_runs):
            model.forward(prompt, budget=budget)
            model.generate(prompt[:1], budget=budget)
            prefill_outputs = model.forward(
                prompt[:1],
                budget=budget,
                cache=DecoderKVCache.create(model.config.num_hidden_layers),
            )
            cache = prefill_outputs.cache
            next_logits = prefill_outputs.logits[:, -1, :]
            if cache is None:
                raise RuntimeError("Prefill cache was unexpectedly None during warmup.")
            for _ in range(args.max_new_tokens):
                next_token = model._sample_next_token(
                    next_logits,
                    temperature=budget.temperature,
                    top_k=budget.top_k,
                )
                decode_outputs = model.forward(next_token[:, None], budget=budget, cache=cache)
                next_logits = decode_outputs.logits[:, -1, :]
                cache = decode_outputs.cache
        synchronize_device(device)
        forward_samples_ms: list[float] = []
        generate_samples_ms: list[float] = []
        cached_decode_samples_ms: list[float] = []
        timed_outputs = None
        timed_generated = None
        timed_decode_shape: list[int] | None = None
        for _ in range(args.timing_runs):
            forward_start = time.perf_counter()
            timed_outputs = model.forward(prompt, budget=budget)
            synchronize_device(device)
            forward_samples_ms.append((time.perf_counter() - forward_start) * 1000.0)
        for _ in range(args.timing_runs):
            generate_start = time.perf_counter()
            timed_generated = model.generate(prompt[:1], budget=budget)
            synchronize_device(device)
            generate_samples_ms.append((time.perf_counter() - generate_start) * 1000.0)
        for _ in range(args.timing_runs):
            cache = DecoderKVCache.create(model.config.num_hidden_layers)
            prefill_outputs = model.forward(prompt[:1], budget=budget, cache=cache)
            cache = prefill_outputs.cache
            next_logits = prefill_outputs.logits[:, -1, :]
            if cache is None:
                raise RuntimeError("Prefill cache was unexpectedly None during decode timing.")
            decoded = prompt[:1]
            synchronize_device(device)
            decode_start = time.perf_counter()
            for _ in range(args.max_new_tokens):
                next_token = model._sample_next_token(
                    next_logits,
                    temperature=budget.temperature,
                    top_k=budget.top_k,
                )
                decoded = torch.cat((decoded, next_token[:, None]), dim=1)
                decode_outputs = model.forward(next_token[:, None], budget=budget, cache=cache)
                next_logits = decode_outputs.logits[:, -1, :]
                cache = decode_outputs.cache
            synchronize_device(device)
            cached_decode_samples_ms.append((time.perf_counter() - decode_start) * 1000.0)
            timed_decode_shape = list(decoded.shape)
        if timed_outputs is None or timed_generated is None:
            raise RuntimeError("Timing runs must be at least 1.")
    forward_stats = latency_summary(forward_samples_ms)
    generate_stats = latency_summary(generate_samples_ms)
    cached_decode_stats = latency_summary(cached_decode_samples_ms)
    generated_new_tokens = max(int(timed_generated.size(1) - prompt[:1].size(1)), 0)
    return {
        "backend": backend,
        "prompt_length": prompt_length,
        "latent_kv_dim": latent_kv_dim,
        "scale_invariant_last_n_layers": scale_invariant_last_n_layers,
        "scale_invariant_decode_mode": scale_invariant_decode_mode if uses_scale_invariant_scores(backend) else None,
        "logits_shape": list(outputs.logits.shape),
        "generated_shape": list(generated.shape),
        "forward_latency": forward_stats,
        "generate_latency": generate_stats,
        "cached_decode_latency": cached_decode_stats,
        "forward_tokens_per_second_median": float(prompt.numel() / max(float(forward_stats["median_ms"]) / 1000.0, 1e-9)),
        "generate_new_tokens_per_second_median": float(
            generated_new_tokens / max(float(generate_stats["median_ms"]) / 1000.0, 1e-9)
        ),
        "cached_decode_new_tokens_per_second_median": float(
            generated_new_tokens / max(float(cached_decode_stats["median_ms"]) / 1000.0, 1e-9)
        ),
        "timed_logits_shape": list(timed_outputs.logits.shape),
        "timed_generated_shape": list(timed_generated.shape),
        "timed_decode_shape": timed_decode_shape,
        "cache": cache_stats(outputs.cache),
    }


def main() -> None:
    args = parse_args()
    device = auto_device()
    torch.manual_seed(args.seed)
    prompt_lengths = prompt_lengths_for_run(args)
    mla_latent_kv_dims = mla_latent_kv_dims_for_run(args)
    scale_invariant_last_n_layers = scale_invariant_last_n_layers_for_run(args)
    scale_invariant_decode_modes = scale_invariant_decode_modes_for_run(args)
    sweep_results: list[dict[str, object]] = []
    for prompt_length in prompt_lengths:
        prompt = torch.randint(0, args.vocab_size, (args.batch_size, prompt_length), device=device)
        for backend in ordered_backends():
            scale_invariant_options = scale_invariant_last_n_layers if uses_scale_invariant_scores(backend) else [None]
            decode_modes = scale_invariant_decode_modes if uses_scale_invariant_scores(backend) else ["all_tokens"]
            for layer_count in scale_invariant_options:
                for decode_mode in decode_modes:
                    if uses_latent_kv(backend):
                        for latent_kv_dim in mla_latent_kv_dims:
                            sweep_results.append(
                                run_backend(
                                    backend=backend,
                                    args=args,
                                    prompt=prompt,
                                    device=device,
                                    prompt_length=prompt_length,
                                    latent_kv_dim=latent_kv_dim,
                                    scale_invariant_last_n_layers=layer_count,
                                    scale_invariant_decode_mode=decode_mode,
                                )
                            )
                        continue
                    sweep_results.append(
                        run_backend(
                            backend=backend,
                            args=args,
                            prompt=prompt,
                            device=device,
                            prompt_length=prompt_length,
                            latent_kv_dim=None,
                            scale_invariant_last_n_layers=layer_count,
                            scale_invariant_decode_mode=decode_mode,
                        )
                    )
    results = {
        "device": device.type,
        "batch_size": args.batch_size,
        "prompt_lengths": prompt_lengths,
        "mla_latent_kv_dims": mla_latent_kv_dims,
        "scale_invariant_last_n_layers": scale_invariant_last_n_layers,
        "scale_invariant_decode_modes": scale_invariant_decode_modes,
        "scale_invariant_tau": args.scale_invariant_tau,
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
