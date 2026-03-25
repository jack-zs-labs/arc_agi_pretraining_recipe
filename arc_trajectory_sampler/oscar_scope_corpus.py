from __future__ import annotations

from dataclasses import dataclass
import html
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable, Sequence


OSCAR_SCOPE_BENCHMARK = "oscar_scope"
OSCAR_SCOPE_VIEW_CHOICES = ("native_chunk", "section_outline")
DEFAULT_OSCAR_SCOPE_VIEWS = OSCAR_SCOPE_VIEW_CHOICES
_SUPPORTED_EXTENSIONS = (".tex", ".md", ".html", ".pdf")
_EXTENSION_PRIORITY = {
    ".tex": 0,
    ".md": 1,
    ".html": 2,
    ".pdf": 3,
}
_INPUT_RE = re.compile(r"\\(?:input|include)\{([^{}]+)\}")
_DOCUMENT_RE = re.compile(r"\\begin\{document\}(.*?)\\end\{document\}", re.S)
_TEX_SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\{(.*?)\}", re.S)
_TEX_ENV_START_RE = re.compile(
    r"\\begin\{(definition|assumption|remark|example|principle|theorem|proposition|lemma|corollary|abstract)\}"
    r"(?:\[(.*?)\])?"
)
_TEX_ENV_END_RE = re.compile(
    r"\\end\{(definition|assumption|remark|example|principle|theorem|proposition|lemma|corollary|abstract)\}"
)
_TEX_FORMATTING_COMMANDS = (
    "textbf",
    "textit",
    "emph",
    "underline",
    "texttt",
    "textrm",
    "textsf",
    "textsc",
    "large",
    "Large",
    "LARGE",
    "small",
    "footnotesize",
    "normalsize",
)
_TEX_TRIVIAL_COMMAND_RE = re.compile(
    r"\\(?:maketitle|tableofcontents|newpage|clearpage|thispagestyle\{.*?\}|pagestyle\{.*?\}|"
    r"fancyhf\{\}|setstretch\{.*?\}|captionsetup\{.*?\}|hypersetup\{.*?\}|"
    r"titleformat\{.*?\}\{.*?\}\{.*?\}\{.*?\}\{.*?\}|"
    r"definecolor\{.*?\}\{.*?\}\{.*?\}|"
    r"theoremstyle\{.*?\}|newtheorem\{.*?\}(?:\{.*?\}){1,2}|"
    r"newcommand\{.*?\}(?:\[[0-9]+\])?\{.*?\}|DeclareMathOperator\{.*?\}\{.*?\})"
)
_TEX_DROP_COMMAND_RE = re.compile(r"\\(?:label|cite|ref|pageref)\{.*?\}")
_TEX_URL_RE = re.compile(r"\\url\{(.*?)\}")
_TEX_HREF_RE = re.compile(r"\\href\{(.*?)\}\{(.*?)\}")
_TEX_FOOTNOTE_RE = re.compile(r"\\footnote\{(.*?)\}", re.S)
_TEX_ITEM_RE = re.compile(r"\\item(?:\s*\[[^\]]*\])?")
_TEX_LINEBREAK_RE = re.compile(r"\\\\")
_TEX_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)", re.S)
_TEX_DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.S)
_TEX_DOLLAR_MATH_RE = re.compile(r"\$(.+?)\$", re.S)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class OscarScopeRecord:
    benchmark: str
    text: str
    document_id: str
    chunk_index: int
    view: str
    metadata: dict[str, object]


class _HTMLToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def discover_oscar_scope_roots() -> tuple[Path, ...]:
    candidates: list[Path] = []
    env_root = os.environ.get("OSCAR_SCOPE_ROOT")
    if env_root:
        for raw_part in env_root.split(os.pathsep):
            part = raw_part.strip()
            if part:
                candidates.append(Path(part).expanduser())
    sibling_root = Path(__file__).resolve().parents[3] / "oscar_design_docs"
    if sibling_root.exists():
        candidates.append(sibling_root)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)


def _candidate_scope_directories(root: Path) -> tuple[Path, ...]:
    normalized = root.resolve()
    if normalized.is_file():
        return ()
    directories = []
    spec_dir = normalized / "spec"
    notes_dir = normalized / "meeting_notes_for_spec"
    if spec_dir.exists():
        directories.append(spec_dir)
    if notes_dir.exists():
        directories.append(notes_dir)
    if directories:
        return tuple(directories)
    return (normalized,)


def _relative_document_key(path: Path, roots: Sequence[Path]) -> str:
    for root in roots:
        try:
            return str(path.resolve().relative_to(root.resolve()).with_suffix(""))
        except ValueError:
            continue
    return path.with_suffix("").name


def resolve_oscar_scope_files(
    *,
    roots: Sequence[str | Path] = (),
    paths: Sequence[str | Path] = (),
    auto_discover: bool = True,
) -> tuple[Path, ...]:
    resolved_roots = [Path(root).expanduser() for root in roots]
    if auto_discover:
        resolved_roots.extend(discover_oscar_scope_roots())
    candidate_roots = [root.resolve() for root in resolved_roots if root.exists()]
    explicit_paths = [Path(path).expanduser().resolve() for path in paths if Path(path).expanduser().exists()]

    grouped: dict[str, Path] = {}
    for explicit_path in explicit_paths:
        if explicit_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        key = _relative_document_key(explicit_path, candidate_roots)
        previous = grouped.get(key)
        if previous is None or _EXTENSION_PRIORITY[explicit_path.suffix.lower()] < _EXTENSION_PRIORITY[previous.suffix.lower()]:
            grouped[key] = explicit_path

    for root in candidate_roots:
        for directory in _candidate_scope_directories(root):
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix not in _SUPPORTED_EXTENSIONS:
                    continue
                key = _relative_document_key(path, (root, directory))
                previous = grouped.get(key)
                if previous is None or _EXTENSION_PRIORITY[suffix] < _EXTENSION_PRIORITY[previous.suffix.lower()]:
                    grouped[key] = path.resolve()
    return tuple(sorted(grouped.values(), key=_document_sort_key))


def _strip_tex_comments(raw_text: str) -> str:
    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line
        comment_index = None
        for index, char in enumerate(line):
            if char != "%":
                continue
            if index > 0 and line[index - 1] == "\\":
                continue
            comment_index = index
            break
        if comment_index is not None:
            line = line[:comment_index]
        lines.append(line)
    return "\n".join(lines)


def _read_tex_with_includes(path: Path, *, seen: set[Path] | None = None) -> str:
    if seen is None:
        seen = set()
    resolved = path.resolve()
    if resolved in seen:
        return ""
    seen.add(resolved)
    raw_text = _strip_tex_comments(resolved.read_text(encoding="utf-8"))

    def replace_include(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        include_path = resolved.parent / target
        if include_path.suffix.lower() != ".tex":
            include_path = include_path.with_suffix(".tex")
        if not include_path.exists():
            return ""
        return _read_tex_with_includes(include_path, seen=seen)

    return _INPUT_RE.sub(replace_include, raw_text)


def _unwrap_tex_command(text: str, command: str) -> str:
    pattern = re.compile(rf"\\{command}\{{([^{{}}]*)\}}")
    previous = None
    current = text
    while previous != current:
        previous = current
        current = pattern.sub(r"\1", current)
    return current


def _cleanup_tex_inline_fragment(text: str) -> str:
    text = _TEX_HREF_RE.sub(lambda item: f"{item.group(2)} ({item.group(1)})", text)
    text = _TEX_URL_RE.sub(lambda item: item.group(1), text)
    text = _TEX_INLINE_MATH_RE.sub(lambda item: item.group(1).strip(), text)
    text = _TEX_DISPLAY_MATH_RE.sub(lambda item: item.group(1).strip(), text)
    text = _TEX_DOLLAR_MATH_RE.sub(lambda item: item.group(1).strip(), text)
    text = re.sub(r"\\\\\s*", " ", text)
    for command in _TEX_FORMATTING_COMMANDS:
        text = _unwrap_tex_command(text, command)
    text = re.sub(r"\\vspace\{.*?\}", " ", text)
    text = re.sub(r"\\(?:large|Large|LARGE|small|normalsize)\b", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("\\\\", " ")
    text = text.replace("\\", " ")
    text = text.replace("{", " ").replace("}", " ")
    return _cleanup_plain_text(text)


def _extract_braced_command_body(text: str, command: str) -> str | None:
    needle = f"\\{command}" + "{"
    start = text.find(needle)
    if start < 0:
        return None
    index = start + len(needle)
    depth = 1
    chunks: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
            chunks.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chunks)
            chunks.append(char)
        else:
            chunks.append(char)
        index += 1
    return None


def _extract_tex_title(raw_text: str, fallback: str) -> str:
    body = _extract_braced_command_body(raw_text, "title")
    if body is not None:
        title = _cleanup_tex_inline_fragment(body).replace("\n", " ")
        title = _cleanup_plain_text(title)
        if title:
            return title
    return fallback


def _cleanup_plain_text(text: str) -> str:
    text = html.unescape(text)
    lines = [_MULTISPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def _tex_environment_heading(match: re.Match[str]) -> str:
    environment = match.group(1).strip()
    label = environment.capitalize()
    qualifier = match.group(2)
    if qualifier:
        label = f"{label} ({_cleanup_plain_text(qualifier)})"
    return f"\n{label}\n"


def extract_text_from_tex(path: Path) -> tuple[str, str]:
    raw_text = _read_tex_with_includes(path)
    title = _extract_tex_title(raw_text, fallback=path.stem.replace("_", " ").strip())
    match = _DOCUMENT_RE.search(raw_text)
    body = match.group(1) if match else raw_text
    body = _TEX_HREF_RE.sub(lambda item: f"{item.group(2)} ({item.group(1)})", body)
    body = _TEX_URL_RE.sub(lambda item: item.group(1), body)
    body = _TEX_FOOTNOTE_RE.sub(lambda item: f" [Footnote: {item.group(1)}] ", body)
    body = _TEX_DISPLAY_MATH_RE.sub(lambda item: f"\n{item.group(1).strip()}\n", body)
    body = _TEX_INLINE_MATH_RE.sub(lambda item: item.group(1).strip(), body)
    body = _TEX_DOLLAR_MATH_RE.sub(lambda item: item.group(1).strip(), body)
    body = _TEX_SECTION_RE.sub(
        lambda item: (
            "\n\n"
            + ("#" if item.group(1) == "section" else "##" if item.group(1) == "subsection" else "###")
            + " "
            + _cleanup_plain_text(item.group(2))
            + "\n\n"
        ),
        body,
    )
    body = _TEX_ENV_START_RE.sub(_tex_environment_heading, body)
    body = _TEX_ENV_END_RE.sub("\n", body)
    body = _TEX_ITEM_RE.sub("\n- ", body)
    body = _TEX_LINEBREAK_RE.sub("\n", body)
    body = _TEX_DROP_COMMAND_RE.sub("", body)
    body = _TEX_TRIVIAL_COMMAND_RE.sub("", body)
    for command in _TEX_FORMATTING_COMMANDS:
        body = _unwrap_tex_command(body, command)
    body = re.sub(r"\\(?:begin|end)\{[^{}]+\}", "\n", body)
    body = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", body)
    body = body.replace("{", " ").replace("}", " ")
    body = f"# {title}\n\n{body}"
    return title, _cleanup_plain_text(body)


def extract_text_from_markdown(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    title = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
    if not title:
        title = path.stem.replace("_", " ").strip()
    cleaned = _cleanup_plain_text(text)
    if not cleaned.startswith("#"):
        cleaned = f"# {title}\n\n{cleaned}"
    return title, cleaned


def extract_text_from_html(path: Path) -> tuple[str, str]:
    raw_html = path.read_text(encoding="utf-8")
    parser = _HTMLToTextParser()
    parser.feed(raw_html)
    text = _cleanup_plain_text(parser.text())
    title_match = re.search(r"<title>(.*?)</title>", raw_html, re.I | re.S)
    title = _cleanup_plain_text(title_match.group(1)) if title_match else path.stem.replace("_", " ").strip()
    if not text.startswith("#"):
        text = f"# {title}\n\n{text}"
    return title, text


def extract_text_from_pdf(path: Path) -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-nopgbrk", "-enc", "UTF-8", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = _cleanup_plain_text(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError):
        text = ""
    if not text:
        title = path.stem.replace("_", " ").strip()
        return title, f"# {title}"
    title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem.replace("_", " ").strip())
    if not text.startswith("#"):
        text = f"# {title}\n\n{text}"
    return title, text


def extract_document_text(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".tex":
        return extract_text_from_tex(path)
    if suffix == ".md":
        return extract_text_from_markdown(path)
    if suffix == ".html":
        return extract_text_from_html(path)
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    title = path.stem.replace("_", " ").strip()
    return title, _cleanup_plain_text(path.read_text(encoding="utf-8"))


def _document_group(path: Path) -> str:
    lowered_parts = [part.lower() for part in path.parts]
    if "meeting_notes_for_spec" in lowered_parts:
        return "meeting_notes"
    if "spec" in lowered_parts:
        return "spec"
    return path.parent.name.lower()


def _document_sort_key(path: Path) -> tuple[int, str, str]:
    group = _document_group(path)
    group_priority = 0 if group == "spec" else 1 if group == "meeting_notes" else 2
    stem = path.stem.lower()
    if stem.startswith("unified_process_intelligence"):
        stem_priority = 0
    elif stem.startswith("oscar_spec"):
        stem_priority = 1
    elif stem.startswith("recursive_hierarchical_addendum"):
        stem_priority = 2
    elif stem.startswith("process_graph_poc_recipe"):
        stem_priority = 3
    elif stem.startswith("pepsico_example") or stem.startswith("genai_workflow_case_study") or stem.startswith("vc_portfolio_case_study"):
        stem_priority = 4
    elif stem.startswith("pe_workflow_design_integrated_coherent") or stem.startswith("pe_workflow_design_integrated"):
        stem_priority = 5
    else:
        stem_priority = 6
    return (group_priority, stem_priority, stem)


def _document_id(path: Path) -> str:
    group = _document_group(path)
    return f"{group}:{path.stem}"


def _collect_outline_entries(text: str) -> tuple[tuple[int, str], ...]:
    entries: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        if title:
            entries.append((level, title))
    return tuple(entries)


def _outline_text(
    *,
    title: str,
    document_id: str,
    group: str,
    source_kind: str,
    entries: Sequence[tuple[int, str]],
) -> str:
    lines = [
        "OSCAR_SCOPE_OUTLINE_V1",
        f"doc_group={group}",
        f"doc_title={title}",
        f"doc_id={document_id}",
        f"source_kind={source_kind}",
        "content_kind=section_outline",
        "",
    ]
    for level, entry_title in entries:
        indent = "  " * max(level - 1, 0)
        lines.append(f"{indent}- {entry_title}")
    return "\n".join(lines).strip() + "\n"


def _native_chunk_text(
    *,
    title: str,
    document_id: str,
    group: str,
    source_kind: str,
    section_path: str,
    body: str,
) -> str:
    lines = [
        "OSCAR_SCOPE_DOC_V1",
        f"doc_group={group}",
        f"doc_title={title}",
        f"doc_id={document_id}",
        f"source_kind={source_kind}",
        f"section_path={section_path}",
        "content_kind=native_chunk",
        "",
        body.strip(),
    ]
    return "\n".join(lines).strip() + "\n"


def _split_long_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = _SENTENCE_BOUNDARY_RE.split(paragraph)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not buffer else f"{buffer} {sentence}"
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(sentence) <= max_chars:
            buffer = sentence
            continue
        for start in range(0, len(sentence), max_chars):
            chunks.append(sentence[start : start + max_chars].strip())
        buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def _chunk_document_sections(
    *,
    title: str,
    text: str,
    max_chars: int,
) -> list[tuple[str, str]]:
    section_stack: list[str] = [title]
    current_lines: list[str] = []
    chunks: list[tuple[str, str]] = []

    def flush_current_lines() -> None:
        nonlocal current_lines
        paragraphs = [
            paragraph.strip()
            for paragraph in "\n".join(current_lines).split("\n\n")
            if paragraph.strip()
        ]
        if not paragraphs:
            current_lines = []
            return
        buffer = ""
        section_path = " > ".join(section_stack)
        for paragraph in paragraphs:
            for paragraph_piece in _split_long_paragraph(paragraph, max_chars=max_chars):
                candidate = paragraph_piece if not buffer else f"{buffer}\n\n{paragraph_piece}"
                if len(candidate) <= max_chars:
                    buffer = candidate
                    continue
                if buffer:
                    chunks.append((section_path, buffer.strip()))
                buffer = paragraph_piece
        if buffer:
            chunks.append((section_path, buffer.strip()))
        current_lines = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            flush_current_lines()
            level = len(match.group(1))
            heading_title = match.group(2).strip()
            section_stack = section_stack[:level]
            section_stack.append(heading_title)
            continue
        current_lines.append(line)
    flush_current_lines()
    return chunks


def build_oscar_scope_records(
    *,
    roots: Sequence[str | Path] = (),
    paths: Sequence[str | Path] = (),
    auto_discover: bool = True,
    max_documents: int | None = None,
    max_chunks: int | None = None,
    views: Sequence[str] = DEFAULT_OSCAR_SCOPE_VIEWS,
    max_chunk_chars: int = 2200,
) -> tuple[OscarScopeRecord, ...]:
    unsupported = sorted(view for view in views if view not in OSCAR_SCOPE_VIEW_CHOICES)
    if unsupported:
        raise ValueError(f"Unsupported Oscar scope views: {unsupported}")
    records: list[OscarScopeRecord] = []
    files = resolve_oscar_scope_files(roots=roots, paths=paths, auto_discover=auto_discover)
    if max_documents is not None:
        files = files[:max_documents]
    per_document_chunk_cap = None
    if max_chunks is not None and files:
        per_document_chunk_cap = max(1, max_chunks // len(files))
    for path in files:
        title, text = extract_document_text(path)
        if not text.strip():
            continue
        document_id = _document_id(path)
        group = _document_group(path)
        outline_entries = _collect_outline_entries(text)
        source_kind = path.suffix.lower().lstrip(".")
        if "section_outline" in views and outline_entries:
            records.append(
                OscarScopeRecord(
                    benchmark=OSCAR_SCOPE_BENCHMARK,
                    text=_outline_text(
                        title=title,
                        document_id=document_id,
                        group=group,
                        source_kind=source_kind,
                        entries=outline_entries,
                    ),
                    document_id=document_id,
                    chunk_index=0,
                    view="section_outline",
                    metadata={
                        "doc_group": group,
                        "doc_title": title,
                        "doc_id": document_id,
                        "source_kind": source_kind,
                        "source_path": str(path),
                        "view": "section_outline",
                        "outline_depth": max((level for level, _title in outline_entries), default=1),
                    },
                )
            )
        if "native_chunk" in views:
            section_chunks = _chunk_document_sections(title=title, text=text, max_chars=max_chunk_chars)
            if per_document_chunk_cap is not None:
                section_chunks = section_chunks[:per_document_chunk_cap]
            for chunk_offset, (section_path, body) in enumerate(section_chunks):
                records.append(
                    OscarScopeRecord(
                        benchmark=OSCAR_SCOPE_BENCHMARK,
                        text=_native_chunk_text(
                            title=title,
                            document_id=document_id,
                            group=group,
                            source_kind=source_kind,
                            section_path=section_path,
                            body=body,
                        ),
                        document_id=document_id,
                        chunk_index=chunk_offset,
                        view="native_chunk",
                        metadata={
                            "doc_group": group,
                            "doc_title": title,
                            "doc_id": document_id,
                            "source_kind": source_kind,
                            "source_path": str(path),
                            "view": "native_chunk",
                            "section_path": section_path,
                            "char_count": len(body),
                        },
                    )
                )
                if max_chunks is not None and len(records) >= max_chunks:
                    return tuple(records[:max_chunks])
    if max_chunks is not None:
        records = records[:max_chunks]
    return tuple(records)


def scope_source_summary(records: Iterable[OscarScopeRecord]) -> dict[str, dict[str, int]]:
    view_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for record in records:
        view_counts[record.view] = view_counts.get(record.view, 0) + 1
        group = str(record.metadata.get("doc_group", "unknown"))
        group_counts[group] = group_counts.get(group, 0) + 1
    return {
        "view_counts": dict(sorted(view_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
    }
