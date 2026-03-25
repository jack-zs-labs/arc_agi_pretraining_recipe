from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This smoke check requires torch. Install requirements-models.txt or use .venv_atari.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import AttentionBackendConfig, DecoderLanguageModel, DecoderModelConfig, InferenceBudget


def auto_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cache_numel(cache: object) -> int:
    if cache is None:
        return 0
    total = 0
    for layer in cache.layers:
        for name in ("key", "value", "latent"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                total += int(tensor.numel())
    return total


def main() -> None:
    device = auto_device()
    torch.manual_seed(0)
    sdpa_config = DecoderModelConfig(
        vocab_size=257,
        max_position_embeddings=256,
        hidden_size=96,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        intermediate_size=256,
        attention=AttentionBackendConfig(backend="sdpa", dropout=0.0),
    )
    model = DecoderLanguageModel(sdpa_config).to(device)
    model.eval()

    prompt = torch.randint(0, sdpa_config.vocab_size, (2, 12), device=device)
    forward_budget = InferenceBudget(
        active_layers=3,
        max_cache_tokens=16,
        use_kv_cache=True,
    )
    outputs = model.forward(prompt, budget=forward_budget)
    assert outputs.logits.shape == (2, 12, sdpa_config.vocab_size)
    assert outputs.last_hidden_state.shape == (2, 12, sdpa_config.hidden_size)
    assert outputs.cache is not None
    assert outputs.cache.tokens_seen == 12
    assert outputs.cache.current_length == 12

    step = prompt[:, -1:]
    cached_outputs = model.forward(step, budget=forward_budget, cache=outputs.cache)
    assert cached_outputs.logits.shape == (2, 1, sdpa_config.vocab_size)
    assert cached_outputs.cache is not None
    assert cached_outputs.cache.tokens_seen == 13
    assert cached_outputs.cache.current_length == 13

    generation_budget = InferenceBudget(
        max_prompt_tokens=10,
        max_new_tokens=6,
        active_layers=2,
        max_cache_tokens=10,
        use_kv_cache=True,
        stop_token_ids=(0,),
    )
    generated = model.generate(prompt[:1], budget=generation_budget)
    expected_prompt_length = min(prompt[:1].size(1), generation_budget.max_prompt_tokens or prompt[:1].size(1))
    assert generated.size(0) == 1
    assert generated.size(1) >= expected_prompt_length
    assert generated.size(1) <= expected_prompt_length + generation_budget.max_new_tokens

    hybrid_config = DecoderModelConfig(
        vocab_size=257,
        max_position_embeddings=512,
        hidden_size=96,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        intermediate_size=256,
        attention=AttentionBackendConfig(
            backend="hybrid",
            dropout=0.0,
            sliding_window=8,
            global_token_stride=4,
            sink_token_count=2,
        ),
    )
    hybrid_model = DecoderLanguageModel(hybrid_config).to(device)
    hybrid_model.eval()
    hybrid_prompt = torch.randint(0, hybrid_config.vocab_size, (1, 20), device=device)
    hybrid_budget = InferenceBudget(
        active_layers=4,
        attention_window=8,
        max_cache_tokens=20,
        use_kv_cache=True,
    )
    hybrid_outputs = hybrid_model.forward(hybrid_prompt, budget=hybrid_budget)
    assert hybrid_outputs.logits.shape == (1, 20, hybrid_config.vocab_size)
    assert hybrid_outputs.cache is not None
    assert hybrid_outputs.cache.current_length == 20
    assert hybrid_outputs.cache.layers[0].positions is not None
    position_tail = hybrid_outputs.cache.layers[0].positions[0, -4:].tolist()
    assert position_tail == [16, 17, 18, 19]

    hybrid_generated = hybrid_model.generate(
        hybrid_prompt[:, :10],
        budget=InferenceBudget(
            max_new_tokens=4,
            active_layers=2,
            attention_window=8,
            max_cache_tokens=12,
            use_kv_cache=True,
        ),
    )
    assert hybrid_generated.size(1) <= 14

    sia_config = DecoderModelConfig(
        vocab_size=257,
        max_position_embeddings=512,
        hidden_size=96,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        intermediate_size=256,
        attention=AttentionBackendConfig(
            backend="sia",
            dropout=0.0,
            scale_invariant_tau=10.0,
        ),
    )
    sia_model = DecoderLanguageModel(sia_config).to(device)
    sia_model.eval()
    sia_prompt = torch.randint(0, sia_config.vocab_size, (1, 20), device=device)
    sia_outputs = sia_model.forward(
        sia_prompt,
        budget=InferenceBudget(
            active_layers=4,
            max_cache_tokens=20,
            use_kv_cache=True,
        ),
    )
    assert sia_outputs.logits.shape == (1, 20, sia_config.vocab_size)
    assert sia_outputs.cache is not None
    assert sia_outputs.cache.current_length == 20

    sia_hybrid_config = DecoderModelConfig(
        vocab_size=257,
        max_position_embeddings=512,
        hidden_size=96,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        intermediate_size=256,
        attention=AttentionBackendConfig(
            backend="sia_hybrid",
            dropout=0.0,
            sliding_window=8,
            global_token_stride=4,
            sink_token_count=2,
            scale_invariant_tau=10.0,
        ),
    )
    sia_hybrid_model = DecoderLanguageModel(sia_hybrid_config).to(device)
    sia_hybrid_model.eval()
    sia_hybrid_prompt = torch.randint(0, sia_hybrid_config.vocab_size, (1, 20), device=device)
    sia_hybrid_outputs = sia_hybrid_model.forward(
        sia_hybrid_prompt,
        budget=InferenceBudget(
            active_layers=4,
            attention_window=8,
            max_cache_tokens=20,
            use_kv_cache=True,
        ),
    )
    assert sia_hybrid_outputs.logits.shape == (1, 20, sia_hybrid_config.vocab_size)
    assert sia_hybrid_outputs.cache is not None
    assert sia_hybrid_outputs.cache.current_length == 20

    mla_config = DecoderModelConfig(
        vocab_size=257,
        max_position_embeddings=512,
        hidden_size=96,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        intermediate_size=256,
        attention=AttentionBackendConfig(
            backend="mla",
            dropout=0.0,
            latent_kv_dim=24,
        ),
    )
    mla_model = DecoderLanguageModel(mla_config).to(device)
    mla_model.eval()
    mla_prompt = torch.randint(0, mla_config.vocab_size, (1, 20), device=device)
    mla_outputs = mla_model.forward(
        mla_prompt,
        budget=InferenceBudget(
            active_layers=4,
            max_cache_tokens=20,
            use_kv_cache=True,
        ),
    )
    assert mla_outputs.logits.shape == (1, 20, mla_config.vocab_size)
    assert mla_outputs.cache is not None
    assert mla_outputs.cache.current_length == 20
    assert mla_outputs.cache.layers[0].latent is not None
    mla_generated = mla_model.generate(
        mla_prompt[:, :10],
        budget=InferenceBudget(
            max_new_tokens=4,
            active_layers=2,
            max_cache_tokens=12,
            use_kv_cache=True,
        ),
    )
    assert mla_generated.size(1) <= 14
    assert cache_numel(mla_outputs.cache) < cache_numel(hybrid_outputs.cache)

    mla_sia_config = DecoderModelConfig(
        vocab_size=257,
        max_position_embeddings=512,
        hidden_size=96,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        intermediate_size=256,
        attention=AttentionBackendConfig(
            backend="mla_sia",
            dropout=0.0,
            latent_kv_dim=24,
            scale_invariant_tau=10.0,
            scale_invariant_last_n_layers=2,
            scale_invariant_decode_mode="prefill_only",
        ),
    )
    mla_sia_model = DecoderLanguageModel(mla_sia_config).to(device)
    mla_sia_model.eval()
    assert mla_sia_model.layers[0].attn.backend.name == "sdpa"
    assert mla_sia_model.layers[1].attn.backend.name == "sdpa"
    assert mla_sia_model.layers[2].attn.backend.name == "sia"
    assert mla_sia_model.layers[3].attn.backend.name == "sia"
    assert mla_sia_model.layers[2].attn.fallback_backend is not None
    assert mla_sia_model.layers[2].attn.fallback_backend.name == "sdpa"
    mla_sia_prompt = torch.randint(0, mla_sia_config.vocab_size, (1, 20), device=device)
    mla_sia_outputs = mla_sia_model.forward(
        mla_sia_prompt,
        budget=InferenceBudget(
            active_layers=4,
            max_cache_tokens=20,
            use_kv_cache=True,
        ),
    )
    assert mla_sia_outputs.logits.shape == (1, 20, mla_sia_config.vocab_size)
    assert mla_sia_outputs.cache is not None
    assert mla_sia_outputs.cache.current_length == 20
    assert mla_sia_outputs.cache.layers[0].latent is not None

    summary = {
        "device": device.type,
        "forward_logits_shape": list(outputs.logits.shape),
        "cached_logits_shape": list(cached_outputs.logits.shape),
        "generated_shape": list(generated.shape),
        "hybrid_forward_logits_shape": list(hybrid_outputs.logits.shape),
        "hybrid_generated_shape": list(hybrid_generated.shape),
        "hybrid_cache_position_tail": position_tail,
        "sia_forward_logits_shape": list(sia_outputs.logits.shape),
        "sia_cache_numel": cache_numel(sia_outputs.cache),
        "sia_hybrid_forward_logits_shape": list(sia_hybrid_outputs.logits.shape),
        "sia_hybrid_cache_numel": cache_numel(sia_hybrid_outputs.cache),
        "mla_forward_logits_shape": list(mla_outputs.logits.shape),
        "mla_generated_shape": list(mla_generated.shape),
        "mla_cache_numel": cache_numel(mla_outputs.cache),
        "mla_sia_forward_logits_shape": list(mla_sia_outputs.logits.shape),
        "mla_sia_cache_numel": cache_numel(mla_sia_outputs.cache),
        "hybrid_cache_numel": cache_numel(hybrid_outputs.cache),
        "cache_tokens_seen": cached_outputs.cache.tokens_seen if cached_outputs.cache is not None else 0,
        "cache_length": cached_outputs.cache.current_length if cached_outputs.cache is not None else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
