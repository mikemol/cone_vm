#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
    return names


def _load_all_list(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = node.value
                    if isinstance(value, (ast.List, ast.Tuple)):
                        items: list[str] = []
                        for elt in value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                items.append(elt.value)
                        return items
    raise SystemExit(f"__all__ not found or not a literal list in {path}")


def main() -> int:
    exports_path = ROOT / "src" / "prism_vm_core" / "exports.py"
    facade_path = ROOT / "src" / "prism_vm_core" / "facade.py"
    types_path = ROOT / "src" / "prism_vm_core" / "types.py"
    repl_path = ROOT / "src" / "prism_cli" / "repl.py"
    errors_path = ROOT / "src" / "prism_core" / "errors.py"

    export_names = _load_all_list(exports_path)
    available = set()
    for path in (facade_path, types_path, repl_path, errors_path):
        available |= _load_module_names(path)

    missing = [name for name in export_names if name not in available]
    if missing:
        print("Missing exports detected:")
        for name in missing:
            print(f"  - {name}")
        return 1
    print(f"export smoke: ok ({len(export_names)} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
