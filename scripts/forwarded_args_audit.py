#!/usr/bin/env python3
"""Audit function parameters used only for forwarding.

Classifies parameter usage as:
  - direct_forward: used as the entire argument / keyword value in a Call
  - derived_forward: used inside an expression that is itself a Call argument
  - non_forward: used outside of any Call argument
  - unused: no Load uses found

This is a best-effort static analysis to surface candidates for config bundling.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class UseInfo:
    direct_forward: int = 0
    derived_forward: int = 0
    non_forward: int = 0

    @property
    def total(self) -> int:
        return self.direct_forward + self.derived_forward + self.non_forward

    def kind(self) -> str:
        if self.total == 0:
            return "unused"
        if self.non_forward:
            return "non_forward"
        if self.derived_forward:
            return "derived_forward"
        return "direct_forward"


class ParentAnnotator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self.visit(child)


def _call_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> tuple[ast.Call | None, bool]:
    """Return (call_node, direct_arg) if node is inside a Call arg."""
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, ast.Call):
            # Direct if child is one of args or keyword value.
            if child in parent.args:
                return parent, True
            if any(child is kw for kw in parent.keywords):
                return parent, True
            # If child is the value of a keyword.
            for kw in parent.keywords:
                if child is kw.value:
                    return parent, True
            # Otherwise, child is likely part of call.func (not forwarding).
            return parent, False
        child = parent
        parent = parents.get(child)
    return None, False


def _is_within_call_arg(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    call, direct = _call_context(node, parents)
    if call is None:
        return False
    if direct:
        return True
    # If not direct, see if node is within an argument expression.
    # Walk upward until the Call; if the first Call encountered is reached
    # through args/keywords, treat as derived_forward.
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, ast.Call):
            # Determine if child is within args/keywords.
            if child in parent.args:
                return True
            for kw in parent.keywords:
                if child is kw or child is kw.value:
                    return True
            return False
        child = parent
        parent = parents.get(child)
    return False


def analyze_file(path: Path) -> dict[str, dict[str, UseInfo]]:
    tree = ast.parse(path.read_text())
    parent = ParentAnnotator()
    parent.visit(tree)
    parents = parent.parents
    results: dict[str, dict[str, UseInfo]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        fn_name = node.name
        args = [
            a.arg
            for a in (
                node.args.posonlyargs
                + node.args.args
                + node.args.kwonlyargs
            )
        ]
        if node.args.vararg:
            args.append(node.args.vararg.arg)
        if node.args.kwarg:
            args.append(node.args.kwarg.arg)
        use_map = {a: UseInfo() for a in args}

        class UseVisitor(ast.NodeVisitor):
            def visit_Name(self, n: ast.Name) -> None:
                if not isinstance(n.ctx, ast.Load):
                    return
                if n.id not in use_map:
                    return
                call, direct = _call_context(n, parents)
                if call is None:
                    use_map[n.id].non_forward += 1
                else:
                    if direct:
                        use_map[n.id].direct_forward += 1
                    else:
                        if _is_within_call_arg(n, parents):
                            use_map[n.id].derived_forward += 1
                        else:
                            use_map[n.id].non_forward += 1

        UseVisitor().visit(node)
        results[fn_name] = use_map
    return results


def iter_paths(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.py")))
        else:
            out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Python files or directories.")
    args = parser.parse_args()
    paths = iter_paths(args.paths)
    for path in paths:
        print(f"# {path}")
        results = analyze_file(path)
        for fn, use_map in results.items():
            forward_only = []
            derived_only = []
            unused = []
            for name, info in use_map.items():
                kind = info.kind()
                if kind == "direct_forward":
                    forward_only.append(name)
                elif kind == "derived_forward":
                    derived_only.append(name)
                elif kind == "unused":
                    unused.append(name)
            if forward_only or derived_only or unused:
                print(f"{fn}:")
                if forward_only:
                    print(f"  direct_forward_only: {sorted(forward_only)}")
                if derived_only:
                    print(f"  derived_forward_only: {sorted(derived_only)}")
                if unused:
                    print(f"  unused: {sorted(unused)}")
        print()


if __name__ == "__main__":
    main()
