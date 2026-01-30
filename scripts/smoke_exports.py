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


def _check_exports(exports_path: Path, source_paths: tuple[Path, ...]) -> list[str]:
    export_names = _load_all_list(exports_path)
    available: set[str] = set()
    for path in source_paths:
        available |= _load_module_names(path)
    return [name for name in export_names if name not in available]


def main() -> int:
    failures: list[tuple[str, list[str]]] = []

    prism_exports = ROOT / "src" / "prism_vm_core" / "exports.py"
    prism_sources = (
        ROOT / "src" / "prism_vm_core" / "facade.py",
        ROOT / "src" / "prism_vm_core" / "types.py",
        ROOT / "src" / "prism_cli" / "repl.py",
        ROOT / "src" / "prism_core" / "errors.py",
    )
    missing = _check_exports(prism_exports, prism_sources)
    if missing:
        failures.append(("prism_vm_core/exports.py", missing))

    ic_facade = ROOT / "src" / "ic_core" / "facade.py"
    ic_types = ROOT / "src" / "ic_core" / "types.py"
    ic_sources = (
        ic_facade,
        ic_types,
    )
    ic_export_names = _load_all_list(ic_facade)
    ic_export_names += [name for name in _load_all_list(ic_types) if name not in ic_export_names]
    available = set()
    for path in ic_sources:
        available |= _load_module_names(path)
    missing = [name for name in ic_export_names if name not in available]
    if missing:
        failures.append(("ic_core (facade/types __all__)", missing))

    if failures:
        print("Missing exports detected:")
        for label, missing_names in failures:
            print(f"- {label}")
            for name in missing_names:
                print(f"  - {name}")
        return 1

    prism_count = len(_load_all_list(prism_exports))
    ic_count = len(ic_export_names)
    print(f"export smoke: ok (prism={prism_count}, ic={ic_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
