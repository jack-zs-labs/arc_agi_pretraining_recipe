from __future__ import annotations

"""Initial loader for the CoRe static-analysis benchmark."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Iterable, Sequence
from urllib.request import urlopen


CORE_DATA_URL = "https://huggingface.co/datasets/lt-asset/CoRe/resolve/main/data.jsonl"
DEFAULT_CORE_DATA_DIR = "arc_trajectory_sampler/data/core"
DEPENDENCY_KIND_CHOICES = ("control", "data", "infoflow")
TRACE_BOOLEAN_OUTPUT_KEYS = {
    "control": "ControlDependence",
    "data": "DataDependence",
    "infoflow": "InformationFlow",
}
SOURCE_OUTPUT_KEYS = {
    "control": "ControlDependenceSources",
    "data": "DataDependenceSources",
    "infoflow": "InfomationFlowSources",
}
_FENCED_CODE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_QUESTION_RE = re.compile(r"\*\*Question\*\*:\s*(.+?)\n\n\*\*Output\*\*:", re.S)
_CODE_LINE_RE = re.compile(r"^\s*(\d+)(\s+)(.*)$")


@dataclass(frozen=True, order=True)
class CoreNodeRef:
    line: int
    name: str | None = None
    use_kind: str | None = None

    def to_jsonable(self) -> dict[str, object]:
        payload: dict[str, object] = {"line": self.line}
        if self.name is not None:
            payload["name"] = self.name
        if self.use_kind is not None:
            payload["use_kind"] = self.use_kind
        return payload

    def to_benchmark_value(self) -> int | list[object]:
        if self.name is None:
            return self.line
        payload: list[object] = [self.name, self.line]
        if self.use_kind is not None:
            payload.append(self.use_kind)
        return payload

    def node_id(self) -> str:
        if self.name is None:
            return f"line:{self.line}"
        suffix = "" if self.use_kind is None else f":{self.use_kind}"
        return f"var:{self.name}@{self.line}{suffix}"


@dataclass(frozen=True)
class CoreCodeLine:
    line: int
    text: str


@dataclass(frozen=True)
class CORERow:
    task_id: str
    dependency_kind: str
    category: str
    dataset: str
    language: str
    prompt: str
    groundtruth: bool
    src: CoreNodeRef
    dst: CoreNodeRef
    funname: str
    start: int
    end: int
    pid: str
    sid: str
    code_file: str
    label_file: str
    metadata: dict[str, object]


def dependency_kind_from_task_id(task_id: str) -> str:
    dependency_kind = task_id.split("_", 1)[0].strip().lower()
    if dependency_kind not in DEPENDENCY_KIND_CHOICES:
        return "control"
    return dependency_kind


def parse_core_node(value: object, transformed: object | None = None) -> CoreNodeRef:
    name: str | None = None
    line: int | None = None
    use_kind: str | None = None
    if isinstance(value, list):
        if value:
            name = str(value[0]) if value[0] not in (None, "") else None
        if len(value) >= 2:
            line = int(value[1])
        if len(value) >= 3 and value[2] not in (None, ""):
            use_kind = str(value[2])
    elif isinstance(value, tuple):
        return parse_core_node(list(value), transformed)
    elif value is not None:
        line = int(value)

    if isinstance(transformed, dict):
        transformed_name = transformed.get("name")
        transformed_line = transformed.get("line")
        transformed_use_kind = transformed.get("use_kind") or transformed.get("use")
        if transformed_name not in (None, ""):
            name = str(transformed_name)
        if transformed_line is not None:
            line = int(transformed_line)
        if transformed_use_kind not in (None, ""):
            use_kind = str(transformed_use_kind)

    if line is None:
        raise ValueError(f"Unable to resolve CoRe node line from value={value!r} transformed={transformed!r}")
    return CoreNodeRef(line=line, name=name, use_kind=use_kind)


@lru_cache(maxsize=4096)
def extract_core_question(prompt: str) -> str:
    match = _QUESTION_RE.search(prompt)
    if not match:
        return ""
    return match.group(1).strip()


@lru_cache(maxsize=4096)
def extract_core_code_lines(prompt: str) -> tuple[CoreCodeLine, ...]:
    blocks = _FENCED_CODE_RE.findall(prompt)
    if not blocks:
        return ()
    last_block = blocks[-1]
    parsed_rows: list[tuple[int, int, str]] = []
    for raw_line in last_block.splitlines():
        match = _CODE_LINE_RE.match(raw_line.rstrip())
        if not match:
            continue
        parsed_rows.append((int(match.group(1)), len(match.group(2)), match.group(3).rstrip()))
    if not parsed_rows:
        return ()
    code_lines: list[CoreCodeLine] = []
    for line_number, gap_width, text in parsed_rows:
        separator_width = max(1, 4 - len(str(line_number)))
        indentation_width = max(gap_width - separator_width, 0)
        code_lines.append(CoreCodeLine(line=line_number, text=(" " * indentation_width) + text))
    return tuple(code_lines)


def ensure_core_file(data_dir: str | Path = DEFAULT_CORE_DATA_DIR) -> Path:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "data.jsonl"
    if path.exists():
        return path
    with urlopen(CORE_DATA_URL) as response:
        path.write_bytes(response.read())
    return path


def load_core_rows(
    *,
    data_dir: str | Path = DEFAULT_CORE_DATA_DIR,
    max_rows: int | None = None,
    languages: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    dependency_kinds: Sequence[str] | None = None,
) -> tuple[CORERow, ...]:
    path = ensure_core_file(data_dir)
    allowed_languages = set(languages) if languages is not None else None
    allowed_categories = set(categories) if categories is not None else None
    allowed_dependency_kinds = set(dependency_kinds) if dependency_kinds is not None else None
    rows: list[CORERow] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            payload = json.loads(line)
            task_id = str(payload["task_id"])
            dependency_kind = dependency_kind_from_task_id(task_id)
            language = str(payload.get("language", ""))
            category = str(payload.get("category", ""))
            if allowed_languages is not None and language not in allowed_languages:
                continue
            if allowed_categories is not None and category not in allowed_categories:
                continue
            if allowed_dependency_kinds is not None and dependency_kind not in allowed_dependency_kinds:
                continue
            rows.append(
                CORERow(
                    task_id=task_id,
                    dependency_kind=dependency_kind,
                    category=category,
                    dataset=str(payload.get("dataset", "")),
                    language=language,
                    prompt=str(payload.get("prompt", "")),
                    groundtruth=bool(payload.get("groundtruth", False)),
                    src=parse_core_node(payload.get("src"), payload.get("src_transformed")),
                    dst=parse_core_node(payload.get("dst"), payload.get("dst_transformed")),
                    funname=str(payload.get("funname", "")),
                    start=int(payload.get("start", 0)),
                    end=int(payload.get("end", 0)),
                    pid=str(payload.get("pid", "")),
                    sid=str(payload.get("sid", "")),
                    code_file=str(payload.get("code_file", "")),
                    label_file=str(payload.get("label_file", "")),
                    metadata={
                        "src_transformed": payload.get("src_transformed"),
                        "dst_transformed": payload.get("dst_transformed"),
                    },
                )
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    return tuple(rows)


def serialize_core_supervision_text(row: CORERow) -> str:
    query_source = row.src.to_jsonable()
    query_target = row.dst.to_jsonable()
    if row.category == "list_source":
        if row.dependency_kind == "infoflow" and not row.groundtruth:
            source_value: object = False
        else:
            source_value = [row.src.to_benchmark_value()] if row.groundtruth else []
        target = {
            SOURCE_OUTPUT_KEYS[row.dependency_kind]: source_value
        }
    else:
        target = {TRACE_BOOLEAN_OUTPUT_KEYS[row.dependency_kind]: bool(row.groundtruth)}
    return (
        "CORE_TASK_V1\n"
        f"dataset=core\n"
        f"dependency_kind={row.dependency_kind}\n"
        f"category={row.category}\n"
        f"language={row.language}\n"
        f"source_dataset={row.dataset}\n"
        f"task_id={row.task_id}\n"
        f"function={row.funname}\n"
        f"query_source={json.dumps(query_source, separators=(',', ':'), sort_keys=True)}\n"
        f"query_target={json.dumps(query_target, separators=(',', ':'), sort_keys=True)}\n"
        "prompt=\n"
        f"{row.prompt.rstrip()}\n"
        f"target_answer={json.dumps(target, separators=(',', ':'), sort_keys=True)}\n"
    )


def iter_core_supervision_texts(rows: Iterable[CORERow]) -> Iterable[str]:
    for row in rows:
        yield serialize_core_supervision_text(row)
