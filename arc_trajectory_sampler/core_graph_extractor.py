from __future__ import annotations

import ast
from dataclasses import dataclass, field
import re
from typing import Iterable, Sequence

from .core_loader import CoreCodeLine


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
ASSIGNMENT_RE = re.compile(r"(?<![=!<>])=(?!=)")
COMPOUND_ASSIGNMENT_RE = re.compile(r"(\+=|-=|\*=|/=|%=|<<=|>>=|&=|\|=|\^=)")
INCREMENT_RE = re.compile(r"(\+\+|--)\s*([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*)\s*(\+\+|--)")
CONTROL_HEAD_RE = re.compile(r"^\s*(if|else\s+if|else|for|while|switch|case|catch|do)\b")
MUTATING_METHOD_NAMES = {
    "add",
    "append",
    "clear",
    "delete",
    "insert",
    "merge",
    "offer",
    "poll",
    "pop",
    "push",
    "put",
    "remove",
    "replace",
    "set",
    "update",
}
MUTATING_METHOD_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(" + "|".join(sorted(MUTATING_METHOD_NAMES)) + r")\s*\("
)
DECLARATION_RE = re.compile(
    r"^\s*(?:final\s+|const\s+|static\s+|public\s+|private\s+|protected\s+|unsigned\s+|signed\s+|long\s+|short\s+|volatile\s+|transient\s+)*"
    r"(?:void|bool|boolean|byte|char|double|float|int|Integer|long|short|String|size_t|auto)\b"
)
KEYWORDS = {
    "abstract",
    "auto",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "false",
    "final",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "include",
    "int",
    "interface",
    "long",
    "main",
    "new",
    "null",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "String",
    "struct",
    "super",
    "switch",
    "this",
    "throw",
    "throws",
    "true",
    "try",
    "typedef",
    "union",
    "unsigned",
    "using",
    "void",
    "volatile",
    "while",
}
PYTHON_CONTROL_NODE_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
)
GRAPH_BACKEND_CHOICES = ("auto", "heuristic", "python_ast")


@dataclass(frozen=True)
class CoreGraphNode:
    node_id: str
    node_type: str
    label: str
    line: int | None = None
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CoreGraphEdge:
    edge_type: str
    source_id: str
    target_id: str
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CoreCodeGraph:
    nodes: tuple[CoreGraphNode, ...]
    edges: tuple[CoreGraphEdge, ...]
    line_node_ids: dict[int, str]
    var_def_node_ids: dict[tuple[str, int], str]
    backend: str = "heuristic"


def line_node_id(line: int) -> str:
    return f"line:{line}"


def var_def_node_id(name: str, line: int) -> str:
    return f"var:{name}@{line}"


def var_use_node_id(name: str, line: int) -> str:
    return f"use:{name}@{line}"


def normalized_identifiers(text: str) -> list[str]:
    return [
        token
        for token in IDENTIFIER_RE.findall(text)
        if token not in KEYWORDS and not token.isupper()
    ]


def strip_comments(text: str) -> str:
    return text.split("//", 1)[0].rstrip()


def is_control_line(text: str) -> bool:
    return CONTROL_HEAD_RE.match(text) is not None


def split_assignment(text: str) -> tuple[str, str] | None:
    match = COMPOUND_ASSIGNMENT_RE.search(text)
    if match:
        operator = match.group(1)
        lhs, rhs = text.split(operator, 1)
        return lhs, rhs
    match = ASSIGNMENT_RE.search(text)
    if match:
        lhs = text[: match.start()]
        rhs = text[match.end() :]
        return lhs, rhs
    return None


def declaration_defined_names(text: str) -> list[str]:
    if DECLARATION_RE.match(text) is None:
        return []
    if "(" in text and ")" in text and text.rstrip().endswith("{"):
        return []
    declaration = text
    if "=" in declaration:
        declaration = declaration.split("=", 1)[0]
    if "(" in declaration:
        declaration = declaration.split("(", 1)[0]
    tokens = normalized_identifiers(declaration)
    if not tokens:
        return []
    return tokens[1:]


def assignment_defined_names(text: str) -> list[str]:
    assignment = split_assignment(text)
    if assignment is None:
        return []
    lhs, _rhs = assignment
    names = normalized_identifiers(lhs)
    if not names:
        return []
    return [names[-1]]


def increment_defined_names(text: str) -> list[str]:
    defined: list[str] = []
    for match in INCREMENT_RE.finditer(text):
        name = match.group(2) or match.group(3)
        if name is not None:
            defined.append(name)
    return defined


def mutating_receiver_names(text: str) -> list[str]:
    defined: list[str] = []
    for match in MUTATING_METHOD_RE.finditer(text):
        receiver = match.group(1)
        if receiver not in defined:
            defined.append(receiver)
    return defined


def defined_names_for_line(text: str) -> list[str]:
    defined: list[str] = []
    for name in declaration_defined_names(text):
        if name not in defined:
            defined.append(name)
    for name in assignment_defined_names(text):
        if name not in defined:
            defined.append(name)
    for name in increment_defined_names(text):
        if name not in defined:
            defined.append(name)
    for name in mutating_receiver_names(text):
        if name not in defined:
            defined.append(name)
    return defined


def condition_text_for_line(text: str) -> str:
    if not is_control_line(text):
        return ""
    start = text.find("(")
    end = text.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start + 1 : end]


def used_names_for_line(text: str, *, defined_names: Sequence[str]) -> list[str]:
    del defined_names
    used: list[str] = []
    assignment = split_assignment(text)
    if assignment is not None:
        _lhs, rhs = assignment
        for token in normalized_identifiers(rhs):
            if token not in used:
                used.append(token)
    condition_text = condition_text_for_line(text)
    if condition_text:
        for token in normalized_identifiers(condition_text):
            if token not in used:
                used.append(token)
    if not used:
        for token in normalized_identifiers(text):
            if token not in used:
                used.append(token)
    return used


def normalize_language(language: str) -> str:
    return language.strip().lower()


def dedupe_graph(
    *,
    nodes: Sequence[CoreGraphNode],
    edges: Sequence[CoreGraphEdge],
    line_node_ids: dict[int, str],
    var_def_node_ids: dict[tuple[str, int], str],
    backend: str,
) -> CoreCodeGraph:
    deduped_nodes: dict[str, CoreGraphNode] = {}
    for node in nodes:
        deduped_nodes[node.node_id] = node
    deduped_edges: dict[tuple[str, str, str, tuple[tuple[str, object], ...]], CoreGraphEdge] = {}
    for edge in edges:
        deduped_edges[
            (
                edge.edge_type,
                edge.source_id,
                edge.target_id,
                tuple(sorted(edge.attributes.items())),
            )
        ] = edge
    return CoreCodeGraph(
        nodes=tuple(deduped_nodes.values()),
        edges=tuple(deduped_edges.values()),
        line_node_ids=line_node_ids,
        var_def_node_ids=var_def_node_ids,
        backend=backend,
    )


def _extract_heuristic_core_code_graph(
    code_lines: Sequence[CoreCodeLine],
    *,
    language: str,
) -> CoreCodeGraph:
    del language  # The current extractor is language-light and operates on C/Java-like syntax.
    nodes: list[CoreGraphNode] = []
    edges: list[CoreGraphEdge] = []
    line_node_ids: dict[int, str] = {}
    var_def_node_ids: dict[tuple[str, int], str] = {}
    last_defs: dict[str, list[str]] = {}
    active_brace_guards: list[tuple[int, int]] = []
    pending_single_line_guards: list[int] = []
    brace_depth = 0

    for index, code_line in enumerate(code_lines):
        raw_text = strip_comments(code_line.text)
        if not raw_text and code_line.text.strip() == "}":
            brace_depth = max(0, brace_depth - 1)
            active_brace_guards = [guard for guard in active_brace_guards if brace_depth > guard[1]]
            continue

        closing_braces = raw_text.count("}")
        opening_braces = raw_text.count("{")
        brace_depth_before = brace_depth
        if closing_braces:
            brace_depth = max(0, brace_depth - closing_braces)
            active_brace_guards = [guard for guard in active_brace_guards if brace_depth > guard[1]]
        brace_depth_before = brace_depth

        line_id = line_node_id(code_line.line)
        line_node_ids[code_line.line] = line_id
        control_flag = is_control_line(raw_text)
        nodes.append(
            CoreGraphNode(
                node_id=line_id,
                node_type="line",
                label=raw_text or code_line.text.rstrip(),
                line=code_line.line,
                attributes={
                    "is_control": control_flag,
                    "brace_depth": brace_depth_before,
                    "backend": "heuristic",
                },
            )
        )

        if index > 0:
            previous_id = line_node_id(code_lines[index - 1].line)
            edges.append(CoreGraphEdge("next_line", previous_id, line_id))

        for guard_line, _guard_depth in active_brace_guards:
            edges.append(
                CoreGraphEdge(
                    "controls_line",
                    line_node_id(guard_line),
                    line_id,
                    attributes={"scope": "brace"},
                )
            )

        if pending_single_line_guards and raw_text not in {"{", "}"}:
            for guard_line in pending_single_line_guards:
                edges.append(
                    CoreGraphEdge(
                        "controls_line",
                        line_node_id(guard_line),
                        line_id,
                        attributes={"scope": "single_line"},
                    )
                )
            pending_single_line_guards = []

        defined_names = defined_names_for_line(raw_text)
        used_names = used_names_for_line(raw_text, defined_names=defined_names)
        condition_names = normalized_identifiers(condition_text_for_line(raw_text))

        for name in defined_names:
            node_id = var_def_node_id(name, code_line.line)
            var_def_node_ids[(name, code_line.line)] = node_id
            nodes.append(
                CoreGraphNode(
                    node_id=node_id,
                    node_type="var_def",
                    label=name,
                    line=code_line.line,
                    attributes={"name": name, "backend": "heuristic"},
                )
            )
            edges.append(CoreGraphEdge("defines", line_id, node_id))

        seen_uses: set[str] = set()
        for name in used_names:
            if name in seen_uses:
                continue
            seen_uses.add(name)
            use_id = var_use_node_id(name, code_line.line)
            nodes.append(
                CoreGraphNode(
                    node_id=use_id,
                    node_type="var_use",
                    label=name,
                    line=code_line.line,
                    attributes={
                        "name": name,
                        "in_condition": name in condition_names,
                        "backend": "heuristic",
                    },
                )
            )
            edges.append(CoreGraphEdge("uses", line_id, use_id))
            for source_id in last_defs.get(name, []):
                edges.append(CoreGraphEdge("reaches_use", source_id, use_id))
                if name in condition_names:
                    edges.append(CoreGraphEdge("condition_use", source_id, line_id, attributes={"name": name}))

        target_defs = [var_def_node_ids[(name, code_line.line)] for name in defined_names]
        for target_id in target_defs:
            target_name = target_id.split(":", 1)[1].split("@", 1)[0]
            for name in used_names:
                for source_id in last_defs.get(name, []):
                    edges.append(
                        CoreGraphEdge(
                            "candidate_data",
                            source_id,
                            target_id,
                            attributes={"via": name, "target_name": target_name},
                        )
                    )

        for name in defined_names:
            last_defs[name] = [var_def_node_ids[(name, code_line.line)]]

        if control_flag:
            if opening_braces > 0:
                active_brace_guards.append((code_line.line, brace_depth_before))
            else:
                pending_single_line_guards.append(code_line.line)

        brace_depth = brace_depth_before + opening_braces

    return dedupe_graph(
        nodes=nodes,
        edges=edges,
        line_node_ids=line_node_ids,
        var_def_node_ids=var_def_node_ids,
        backend="heuristic",
    )


def _ordered_unique(names: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for name in names:
        if name in KEYWORDS or name not in ordered:
            if name in KEYWORDS:
                continue
            ordered.append(name)
    return ordered


def _python_source(code_lines: Sequence[CoreCodeLine]) -> tuple[str, dict[int, int]]:
    ordered_lines = sorted(code_lines, key=lambda line: line.line)
    source = "\n".join(code_line.text for code_line in ordered_lines)
    line_number_map = {index + 1: code_line.line for index, code_line in enumerate(ordered_lines)}
    return source, line_number_map


def _target_names_from_ast(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in node.elts:
            names.extend(_target_names_from_ast(element))
        return _ordered_unique(names)
    if isinstance(node, ast.Starred):
        return _target_names_from_ast(node.value)
    if isinstance(node, ast.Attribute):
        return _target_names_from_ast(node.value)
    if isinstance(node, ast.Subscript):
        return _target_names_from_ast(node.value)
    return []


def _import_defined_names(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, ast.Import):
        return _ordered_unique(alias.asname or alias.name.split(".", 1)[0] for alias in stmt.names)
    if isinstance(stmt, ast.ImportFrom):
        return _ordered_unique(alias.asname or alias.name for alias in stmt.names)
    return []


def _function_argument_names(arguments: ast.arguments) -> list[str]:
    names: list[str] = []
    for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
        names.append(arg.arg)
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return _ordered_unique(names)


def _load_names_in_ast(nodes: Sequence[ast.AST]) -> list[str]:
    names: list[str] = []
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                names.append(child.id)
    return _ordered_unique(names)


def _mutating_receiver_names_from_ast(nodes: Sequence[ast.AST]) -> list[str]:
    names: list[str] = []
    for node in nodes:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                continue
            if child.func.attr not in MUTATING_METHOD_NAMES:
                continue
            names.extend(_target_names_from_ast(child.func.value))
    return _ordered_unique(names)


def _python_defined_names_for_stmt(stmt: ast.stmt) -> list[str]:
    names: list[str] = []
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        names.append(stmt.name)
        names.extend(_function_argument_names(stmt.args))
    elif isinstance(stmt, ast.ClassDef):
        names.append(stmt.name)
    elif isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            names.extend(_target_names_from_ast(target))
    elif isinstance(stmt, ast.AnnAssign):
        names.extend(_target_names_from_ast(stmt.target))
    elif isinstance(stmt, ast.AugAssign):
        names.extend(_target_names_from_ast(stmt.target))
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        names.extend(_target_names_from_ast(stmt.target))
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            names.extend(_target_names_from_ast(item.optional_vars))
    elif isinstance(stmt, ast.ExceptHandler) and isinstance(stmt.name, str):
        names.append(stmt.name)
    names.extend(_import_defined_names(stmt))
    if isinstance(stmt, ast.Expr):
        names.extend(_mutating_receiver_names_from_ast([stmt.value]))
    return _ordered_unique(names)


def _python_used_names_for_stmt(stmt: ast.stmt) -> list[str]:
    expression_nodes: list[ast.AST] = []
    if isinstance(stmt, ast.Assign):
        expression_nodes.append(stmt.value)
    elif isinstance(stmt, ast.AnnAssign):
        if stmt.value is not None:
            expression_nodes.append(stmt.value)
    elif isinstance(stmt, ast.AugAssign):
        expression_nodes.extend([stmt.target, stmt.value])
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        expression_nodes.append(stmt.iter)
    elif isinstance(stmt, (ast.If, ast.While, ast.Assert)):
        expression_nodes.append(stmt.test)
        if isinstance(stmt, ast.Assert) and stmt.msg is not None:
            expression_nodes.append(stmt.msg)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        expression_nodes.extend(item.context_expr for item in stmt.items)
    elif isinstance(stmt, ast.Return) and stmt.value is not None:
        expression_nodes.append(stmt.value)
    elif isinstance(stmt, ast.Expr):
        expression_nodes.append(stmt.value)
    elif isinstance(stmt, ast.Raise):
        if stmt.exc is not None:
            expression_nodes.append(stmt.exc)
        if stmt.cause is not None:
            expression_nodes.append(stmt.cause)
    elif isinstance(stmt, ast.Match):
        expression_nodes.append(stmt.subject)
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        expression_nodes.extend(stmt.decorator_list)
        if stmt.returns is not None:
            expression_nodes.append(stmt.returns)
    elif isinstance(stmt, ast.ClassDef):
        expression_nodes.extend(stmt.bases)
        expression_nodes.extend(stmt.decorator_list)
    elif isinstance(stmt, ast.ExceptHandler) and stmt.type is not None:
        expression_nodes.append(stmt.type)
    return _load_names_in_ast(expression_nodes)


def _python_condition_names_for_stmt(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, (ast.If, ast.While)):
        return _load_names_in_ast([stmt.test])
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return _load_names_in_ast([stmt.iter])
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _load_names_in_ast([item.context_expr for item in stmt.items])
    if isinstance(stmt, ast.Match):
        return _load_names_in_ast([stmt.subject])
    return []


def _python_statement_blocks(stmt: ast.stmt) -> list[tuple[str, Sequence[ast.stmt]]]:
    if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
        return [
            ("body", stmt.body),
            ("orelse", stmt.orelse),
        ]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [("body", stmt.body)]
    if isinstance(stmt, ast.Try):
        blocks: list[tuple[str, Sequence[ast.stmt]]] = [
            ("body", stmt.body),
            ("orelse", stmt.orelse),
            ("finalbody", stmt.finalbody),
        ]
        blocks.extend(("except", handler.body) for handler in stmt.handlers)
        return blocks
    if isinstance(stmt, ast.Match):
        return [("case", case.body) for case in stmt.cases]
    return []


def _python_block_statement_lines(block: Sequence[ast.stmt], line_number_map: dict[int, int]) -> list[int]:
    lines: list[int] = []
    for stmt in block:
        line_number = line_number_map.get(getattr(stmt, "lineno", -1))
        if line_number is not None and line_number not in lines:
            lines.append(line_number)
        for _block_name, nested in _python_statement_blocks(stmt):
            for nested_line in _python_block_statement_lines(nested, line_number_map):
                if nested_line not in lines:
                    lines.append(nested_line)
    return lines


def _extract_python_ast_core_code_graph(code_lines: Sequence[CoreCodeLine]) -> CoreCodeGraph | None:
    if not code_lines:
        return None
    source, line_number_map = _python_source(code_lines)
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return None

    statement_nodes = sorted(
        [
            stmt
            for stmt in ast.walk(tree)
            if isinstance(stmt, ast.stmt) and hasattr(stmt, "lineno")
        ],
        key=lambda stmt: (int(getattr(stmt, "lineno", 0)), int(getattr(stmt, "col_offset", 0))),
    )
    control_lines = {
        line_number_map[int(getattr(stmt, "lineno", -1))]
        for stmt in statement_nodes
        if isinstance(stmt, PYTHON_CONTROL_NODE_TYPES) and int(getattr(stmt, "lineno", -1)) in line_number_map
    }
    statement_types_by_line: dict[int, list[str]] = {}
    for stmt in statement_nodes:
        source_line = int(getattr(stmt, "lineno", -1))
        if source_line not in line_number_map:
            continue
        line_number = line_number_map[source_line]
        statement_types_by_line.setdefault(line_number, [])
        statement_name = type(stmt).__name__
        if statement_name not in statement_types_by_line[line_number]:
            statement_types_by_line[line_number].append(statement_name)

    nodes: list[CoreGraphNode] = []
    edges: list[CoreGraphEdge] = []
    line_node_ids: dict[int, str] = {}
    var_def_node_ids: dict[tuple[str, int], str] = {}
    last_defs: dict[str, list[str]] = {}
    ordered_lines = sorted(code_lines, key=lambda line: line.line)

    for index, code_line in enumerate(ordered_lines):
        line_id = line_node_id(code_line.line)
        line_node_ids[code_line.line] = line_id
        nodes.append(
            CoreGraphNode(
                node_id=line_id,
                node_type="line",
                label=code_line.text.rstrip(),
                line=code_line.line,
                attributes={
                    "is_control": code_line.line in control_lines,
                    "backend": "python_ast",
                    "stmt_types": statement_types_by_line.get(code_line.line, []),
                },
            )
        )
        if index > 0:
            edges.append(CoreGraphEdge("next_line", line_node_id(ordered_lines[index - 1].line), line_id))

    for stmt in statement_nodes:
        source_line = int(getattr(stmt, "lineno", -1))
        if source_line not in line_number_map:
            continue
        line_number = line_number_map[source_line]
        line_id = line_node_id(line_number)

        for block_name, block in _python_statement_blocks(stmt):
            for controlled_line in _python_block_statement_lines(block, line_number_map):
                if controlled_line == line_number:
                    continue
                edges.append(
                    CoreGraphEdge(
                        "controls_line",
                        line_id,
                        line_node_id(controlled_line),
                        attributes={"scope": "ast", "block": block_name},
                    )
                )

        defined_names = _python_defined_names_for_stmt(stmt)
        used_names = _python_used_names_for_stmt(stmt)
        condition_names = _python_condition_names_for_stmt(stmt)

        for name in defined_names:
            node_id = var_def_node_id(name, line_number)
            var_def_node_ids[(name, line_number)] = node_id
            nodes.append(
                CoreGraphNode(
                    node_id=node_id,
                    node_type="var_def",
                    label=name,
                    line=line_number,
                    attributes={"name": name, "backend": "python_ast"},
                )
            )
            edges.append(CoreGraphEdge("defines", line_id, node_id))

        target_def_ids = [var_def_node_ids[(name, line_number)] for name in defined_names]
        for name in used_names:
            use_id = var_use_node_id(name, line_number)
            nodes.append(
                CoreGraphNode(
                    node_id=use_id,
                    node_type="var_use",
                    label=name,
                    line=line_number,
                    attributes={
                        "name": name,
                        "in_condition": name in condition_names,
                        "backend": "python_ast",
                    },
                )
            )
            edges.append(CoreGraphEdge("uses", line_id, use_id))
            for source_id in last_defs.get(name, []):
                edges.append(CoreGraphEdge("reaches_use", source_id, use_id))
                if name in condition_names:
                    edges.append(
                        CoreGraphEdge(
                            "condition_use",
                            source_id,
                            line_id,
                            attributes={"name": name, "backend": "python_ast"},
                        )
                    )

        for target_id in target_def_ids:
            target_name = target_id.split(":", 1)[1].split("@", 1)[0]
            for name in used_names:
                for source_id in last_defs.get(name, []):
                    edges.append(
                        CoreGraphEdge(
                            "candidate_data",
                            source_id,
                            target_id,
                            attributes={
                                "via": name,
                                "target_name": target_name,
                                "backend": "python_ast",
                            },
                        )
                    )

        for name in defined_names:
            last_defs[name] = [var_def_node_ids[(name, line_number)]]

    return dedupe_graph(
        nodes=nodes,
        edges=edges,
        line_node_ids=line_node_ids,
        var_def_node_ids=var_def_node_ids,
        backend="python_ast",
    )


def extract_core_code_graph(
    code_lines: Sequence[CoreCodeLine],
    *,
    language: str,
    backend: str = "auto",
) -> CoreCodeGraph:
    if backend not in GRAPH_BACKEND_CHOICES:
        raise ValueError(f"Unsupported CoRe graph backend: {backend!r}")
    normalized_language = normalize_language(language)
    if backend in {"auto", "python_ast"} and normalized_language == "python":
        parsed_graph = _extract_python_ast_core_code_graph(code_lines)
        if parsed_graph is not None:
            return parsed_graph
    return _extract_heuristic_core_code_graph(code_lines, language=language)


def graph_nodes_json(nodes: Iterable[CoreGraphNode]) -> list[dict[str, object]]:
    return [
        {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "label": node.label,
            "line": node.line,
            "attributes": node.attributes,
        }
        for node in nodes
    ]


def graph_edges_json(edges: Iterable[CoreGraphEdge]) -> list[dict[str, object]]:
    return [
        {
            "type": edge.edge_type,
            "from": edge.source_id,
            "to": edge.target_id,
            "attributes": edge.attributes,
        }
        for edge in edges
    ]


def cfg_edges_for_graph(graph: CoreCodeGraph) -> tuple[CoreGraphEdge, ...]:
    allowed = {"next_line", "controls_line"}
    return tuple(edge for edge in graph.edges if edge.edge_type in allowed)


def dfg_edges_for_graph(graph: CoreCodeGraph) -> tuple[CoreGraphEdge, ...]:
    allowed = {"defines", "uses", "reaches_use", "candidate_data"}
    return tuple(edge for edge in graph.edges if edge.edge_type in allowed)


def ifg_edges_for_graph(graph: CoreCodeGraph) -> tuple[CoreGraphEdge, ...]:
    allowed = {"candidate_data", "condition_use", "controls_line", "reaches_use"}
    return tuple(edge for edge in graph.edges if edge.edge_type in allowed)
