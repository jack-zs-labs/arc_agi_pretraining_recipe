from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


TOKENIZER_CHOICES: tuple[str, ...] = ("epiplex", "rust_bpe", "byte")
EPIPLEX_TASK_CHOICES: tuple[str, ...] = ("generic", "gsm8k", "math", "reasoning_graph", "tinystories")
BYTE_EOS_TOKEN_ID = 256
BYTE_VOCAB_SIZE = 257
RUST_BPE_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")


class ReasoningTokenizer:
    kind: str
    vocab_size: int
    eos_token_id: int
    pad_token_id: int
    window_pad_token_id: int
    bos_token_id: int | None
    unk_token_id: int | None

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        raise NotImplementedError

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[list[int]]:
        return [
            self.encode(text, add_bos=add_bos, add_eos=add_eos)
            for text in texts
        ]

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def summary(self) -> dict[str, object]:
        raise NotImplementedError


class ByteReasoningTokenizer(ReasoningTokenizer):
    def __init__(self) -> None:
        self.kind = "byte"
        self.vocab_size = BYTE_VOCAB_SIZE
        self.eos_token_id = BYTE_EOS_TOKEN_ID
        self.pad_token_id = BYTE_EOS_TOKEN_ID
        self.window_pad_token_id = BYTE_EOS_TOKEN_ID
        self.bos_token_id = None
        self.unk_token_id = None

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        if add_bos:
            raise ValueError("ByteReasoningTokenizer does not define a BOS token.")
        tokens = list(text.encode("utf-8", errors="strict"))
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, ids: Sequence[int]) -> str:
        payload = bytes(token for token in ids if 0 <= token < BYTE_EOS_TOKEN_ID)
        return payload.decode("utf-8", errors="strict")

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "window_pad_token_id": self.window_pad_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
        }


class EpiplexReasoningTokenizer(ReasoningTokenizer):
    def __init__(
        self,
        *,
        raw_tokenizer: Any,
        module_path: Path,
        task: str,
        requested_vocab_size: int,
        fitted_from_text_count: int,
        load_path: str | None,
        save_path: str | None,
    ) -> None:
        self.kind = "epiplex"
        self.raw_tokenizer = raw_tokenizer
        self.module_path = module_path
        self.task = task
        self.requested_vocab_size = requested_vocab_size
        self.fitted_from_text_count = fitted_from_text_count
        self.load_path = load_path
        self.save_path = save_path
        self.vocab_size = len(self.raw_tokenizer.vocab)
        self.pad_token_id = int(self.raw_tokenizer.vocab["<pad>"])
        self.bos_token_id = int(self.raw_tokenizer.vocab["<bos>"])
        self.eos_token_id = int(self.raw_tokenizer.vocab["<eos>"])
        self.unk_token_id = int(self.raw_tokenizer.vocab["<unk>"])
        # The existing fixed-window trainers pad with EOS semantics, so keep that
        # behavior stable while moving off the byte vocabulary.
        self.window_pad_token_id = self.eos_token_id

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        return self.raw_tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[list[int]]:
        raw_encode_batch = getattr(self.raw_tokenizer, "encode_batch", None)
        if callable(raw_encode_batch):
            return raw_encode_batch(list(texts), add_bos=add_bos, add_eos=add_eos)
        return super().encode_batch(texts, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.raw_tokenizer.decode(ids))

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "task": self.task,
            "requested_vocab_size": self.requested_vocab_size,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "window_pad_token_id": self.window_pad_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
            "num_merges": len(self.raw_tokenizer.merges),
            "base_alphabet_size": len(self.raw_tokenizer.base_alphabet),
            "fitted_from_text_count": self.fitted_from_text_count,
            "load_path": self.load_path,
            "save_path": self.save_path,
            "source_module": str(self.module_path),
        }


class RustBPETokenizer(ReasoningTokenizer):
    def __init__(
        self,
        *,
        raw_tokenizer: Any,
        task: str,
        requested_vocab_size: int,
        fitted_from_text_count: int,
        load_path: str | None,
        save_path: str | None,
    ) -> None:
        self.kind = "rust_bpe"
        self.raw_tokenizer = raw_tokenizer
        self.task = task
        self.requested_vocab_size = requested_vocab_size
        self.fitted_from_text_count = fitted_from_text_count
        self.load_path = load_path
        self.save_path = save_path
        self.vocab_size = int(self.raw_tokenizer.get_vocab_size())
        self.pad_token_id = self._require_special_token_id("<pad>")
        self.bos_token_id = self._require_special_token_id("<bos>")
        self.eos_token_id = self._require_special_token_id("<eos>")
        self.unk_token_id = self._require_special_token_id("<unk>")
        self.window_pad_token_id = self.eos_token_id

    def _require_special_token_id(self, token: str) -> int:
        token_id = self.raw_tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Rust BPE tokenizer is missing required special token {token!r}.")
        return int(token_id)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = [int(token_id) for token_id in self.raw_tokenizer.encode(text).ids]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[list[int]]:
        encodings = self.raw_tokenizer.encode_batch(list(texts))
        token_batches: list[list[int]] = []
        for encoding in encodings:
            ids = [int(token_id) for token_id in encoding.ids]
            if add_bos:
                ids.insert(0, self.bos_token_id)
            if add_eos:
                ids.append(self.eos_token_id)
            token_batches.append(ids)
        return token_batches

    def decode(self, ids: Sequence[int]) -> str:
        filtered = [
            int(token_id)
            for token_id in ids
            if int(token_id) not in {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }
        ]
        return str(self.raw_tokenizer.decode(filtered, skip_special_tokens=False))

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "backend": "huggingface_tokenizers_rust",
            "task": self.task,
            "requested_vocab_size": self.requested_vocab_size,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "window_pad_token_id": self.window_pad_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
            "fitted_from_text_count": self.fitted_from_text_count,
            "load_path": self.load_path,
            "save_path": self.save_path,
            "special_tokens": list(RUST_BPE_SPECIAL_TOKENS),
        }


def add_tokenizer_cli_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_kind: str = "epiplex",
    default_vocab_size: int = 4096,
    default_task: str = "generic",
    default_fit_verbose: bool = False,
) -> None:
    parser.add_argument("--tokenizer", choices=TOKENIZER_CHOICES, default=default_kind)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=default_vocab_size)
    parser.add_argument("--tokenizer-task", choices=EPIPLEX_TASK_CHOICES, default=default_task)
    parser.add_argument("--tokenizer-min-freq", type=int, default=2)
    parser.add_argument("--tokenizer-candidate-pool-size", type=int, default=2048)
    parser.add_argument("--tokenizer-max-piece-chars", type=int, default=24)
    parser.add_argument("--tokenizer-fit-workers", type=int, default=1)
    parser.add_argument("--tokenizer-load", type=str, default="")
    parser.add_argument("--tokenizer-save", type=str, default="")
    parser.add_argument("--tokenizer-fit-verbose", action=argparse.BooleanOptionalAction, default=default_fit_verbose)


def _load_hf_tokenizers() -> dict[str, Any]:
    try:
        from tokenizers import Tokenizer
        from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'tokenizers' package is required for --tokenizer rust_bpe. "
            "Install it from requirements.txt or requirements-models.txt."
        ) from exc
    return {
        "Tokenizer": Tokenizer,
        "decoders": decoders,
        "models": models,
        "normalizers": normalizers,
        "pre_tokenizers": pre_tokenizers,
        "trainers": trainers,
    }


@lru_cache(maxsize=1)
def _load_epiplex_module() -> tuple[Any, Path]:
    here = Path(__file__).resolve()
    roots: list[Path] = []
    override = os.environ.get("EPIPLEX_TOKENIZER_ROOT", "").strip()
    if override:
        roots.append(Path(override))
    roots.extend(
        [
            here.parents[3] / "research" / "exact_learning" / "epi_token",
            here.parents[2] / "research" / "exact_learning" / "epi_token",
            here.parents[1] / "external" / "epi_token",
        ]
    )
    tried: list[str] = []
    for root in roots:
        module_path = root / "epiplex_tokenizer_benchmark.py"
        tried.append(str(module_path))
        if not module_path.exists():
            continue
        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, module_path
    raise FileNotFoundError(
        "Could not locate epiplex_tokenizer_benchmark.py. "
        f"Tried: {tried}. Set EPIPLEX_TOKENIZER_ROOT if the tokenizer lives elsewhere."
    )


def build_reasoning_tokenizer(
    texts: Sequence[str],
    *,
    kind: str,
    vocab_size: int,
    task: str,
    min_freq: int,
    candidate_pool_size: int,
    max_piece_chars: int,
    fit_workers: int,
    fit_verbose: bool,
    load_path: str = "",
    save_path: str = "",
) -> ReasoningTokenizer:
    if kind == "byte":
        return ByteReasoningTokenizer()
    if kind == "rust_bpe":
        modules = _load_hf_tokenizers()
        Tokenizer = modules["Tokenizer"]
        decoders = modules["decoders"]
        models = modules["models"]
        normalizers = modules["normalizers"]
        pre_tokenizers = modules["pre_tokenizers"]
        trainers = modules["trainers"]

        resolved_load_path = load_path or None
        resolved_save_path = save_path or None
        if fit_workers > 0:
            os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, fit_workers)))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        if resolved_load_path:
            raw_tokenizer = Tokenizer.from_file(str(Path(resolved_load_path).resolve()))
            fitted_from_text_count = len(texts)
        else:
            if not texts:
                raise ValueError("Cannot fit a Rust BPE tokenizer without training texts.")
            raw_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
            raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            raw_tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=max(int(vocab_size), len(RUST_BPE_SPECIAL_TOKENS)),
                min_frequency=max(int(min_freq), 1),
                show_progress=fit_verbose,
                special_tokens=list(RUST_BPE_SPECIAL_TOKENS),
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                max_token_length=max(int(max_piece_chars), 1),
            )
            raw_tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
            fitted_from_text_count = len(texts)
            if resolved_save_path:
                save_target = Path(resolved_save_path)
                save_target.parent.mkdir(parents=True, exist_ok=True)
                raw_tokenizer.save(str(save_target))
        return RustBPETokenizer(
            raw_tokenizer=raw_tokenizer,
            task=task,
            requested_vocab_size=vocab_size,
            fitted_from_text_count=fitted_from_text_count,
            load_path=resolved_load_path,
            save_path=resolved_save_path,
        )

    module, module_path = _load_epiplex_module()
    resolved_load_path = load_path or None
    resolved_save_path = save_path or None
    if resolved_load_path:
        raw_tokenizer = module.GreedyBPETokenizer.load(resolved_load_path)
        fitted_from_text_count = len(texts)
    else:
        if not texts:
            raise ValueError("Cannot fit an Epiplex tokenizer without training texts.")
        raw_tokenizer = module.GreedyBPETokenizer(
            vocab_size=vocab_size,
            min_freq=min_freq,
            candidate_pool_size=candidate_pool_size,
            max_piece_chars=max_piece_chars,
            task=task,
            num_workers=max(1, fit_workers),
            verbose=fit_verbose,
        ).fit(texts)
        fitted_from_text_count = len(texts)
        if resolved_save_path:
            save_target = Path(resolved_save_path)
            save_target.parent.mkdir(parents=True, exist_ok=True)
            raw_tokenizer.save(save_target)
    return EpiplexReasoningTokenizer(
        raw_tokenizer=raw_tokenizer,
        module_path=module_path,
        task=task,
        requested_vocab_size=vocab_size,
        fitted_from_text_count=fitted_from_text_count,
        load_path=resolved_load_path,
        save_path=resolved_save_path,
    )


def resolved_tokenizer_load_path(
    tokenizer_summary: Mapping[str, object],
    *,
    manifest_dir: str | Path | None = None,
) -> str:
    candidate = str(
        tokenizer_summary.get("load_path")
        or tokenizer_summary.get("save_path")
        or ""
    ).strip()
    if not candidate:
        return ""
    path = Path(candidate)
    if not path.is_absolute() and manifest_dir is not None:
        path = Path(manifest_dir) / path
    return str(path.resolve())


def load_reasoning_tokenizer_from_summary(
    tokenizer_summary: Mapping[str, object],
    *,
    manifest_dir: str | Path | None = None,
) -> ReasoningTokenizer:
    kind = str(tokenizer_summary["kind"])
    load_path = resolved_tokenizer_load_path(tokenizer_summary, manifest_dir=manifest_dir)
    return build_reasoning_tokenizer(
        (),
        kind=kind,
        vocab_size=int(tokenizer_summary.get("requested_vocab_size", tokenizer_summary["vocab_size"])),
        task=str(tokenizer_summary.get("task", "generic")),
        min_freq=2,
        candidate_pool_size=2048,
        max_piece_chars=24,
        fit_workers=1,
        fit_verbose=False,
        load_path=load_path,
        save_path="",
    )
