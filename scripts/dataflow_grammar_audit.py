#!/usr/bin/env python3
"""Infer forwarding-based parameter bundles and propagate them across calls.

This script performs a two-stage analysis:
  1) Local grouping: within a function, parameters used *only* as direct
     call arguments are grouped by identical forwarding signatures.
  2) Propagation: if a function f calls g, and g has local bundles, then
     f's parameters passed into g's bundled positions are linked as a
     candidate bundle. This is iterated to a fixed point.

The goal is to surface "dataflow grammar" candidates for config dataclasses.

It can also emit a DOT graph (see --dot) so downstream tooling can render
bundle candidates as a dependency graph.
"""
from __future__ import annotations

import argparse
import ast
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import re


@dataclass
class ParamUse:
    direct_forward: set[tuple[str, str]]
    non_forward: bool


class ParentAnnotator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self.visit(child)


def _call_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> tuple[ast.Call | None, bool]:
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, ast.Call):
            if child in parent.args:
                return parent, True
            for kw in parent.keywords:
                if child is kw or child is kw.value:
                    return parent, True
            return parent, False
        child = parent
        parent = parents.get(child)
    return None, False


def _callee_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except Exception:
        return "<call>"


def _iter_paths(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.py")))
        else:
            out.append(path)
    return out


def _collect_functions(tree: ast.AST):
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return funcs


def _param_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = (
        fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    )
    names = [a.arg for a in args]
    if fn.args.vararg:
        names.append(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.append(fn.args.kwarg.arg)
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    return names


def _analyze_function(fn, parents):
    params = _param_names(fn)
    use_map = {p: ParamUse(set(), False) for p in params}
    call_args: list[tuple[str, dict[str, str], dict[str, str]]] = []

    class UseVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            callee = _callee_name(node)
            pos_map = {}
            kw_map = {}
            for idx, arg in enumerate(node.args):
                if isinstance(arg, ast.Name) and arg.id in use_map:
                    pos_map[str(idx)] = arg.id
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if isinstance(kw.value, ast.Name) and kw.value.id in use_map:
                    kw_map[kw.arg] = kw.value.id
            call_args.append((callee, pos_map, kw_map))
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if not isinstance(node.ctx, ast.Load):
                return
            if node.id not in use_map:
                return
            call, direct = _call_context(node, parents)
            if call is None or not direct:
                use_map[node.id].non_forward = True
                return
            callee = _callee_name(call)
            # Determine arg slot.
            slot = None
            for idx, arg in enumerate(call.args):
                if arg is node:
                    slot = f"arg[{idx}]"
                    break
            if slot is None:
                for kw in call.keywords:
                    if kw.value is node and kw.arg is not None:
                        slot = f"kw[{kw.arg}]"
                        break
            if slot is None:
                slot = "arg[?]"
            use_map[node.id].direct_forward.add((callee, slot))

    UseVisitor().visit(fn)
    return use_map, call_args


def _group_by_signature(use_map: dict[str, ParamUse]) -> list[set[str]]:
    sig_map: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    for name, info in use_map.items():
        if info.non_forward:
            continue
        sig = tuple(sorted(info.direct_forward))
        sig_map[sig].append(name)
    groups = [set(names) for names in sig_map.values() if len(names) > 1]
    return groups


def _union_groups(groups: list[set[str]]) -> list[set[str]]:
    changed = True
    while changed:
        changed = False
        out = []
        while groups:
            base = groups.pop()
            merged = True
            while merged:
                merged = False
                for i, other in enumerate(groups):
                    if base & other:
                        base |= other
                        groups.pop(i)
                        merged = True
                        changed = True
                        break
            out.append(base)
        groups = out
    return groups


def _propagate_groups(
    fn_name: str,
    fn_params: list[str],
    call_args,
    callee_groups: dict[str, list[set[str]]],
    callee_param_orders: dict[str, list[str]],
) -> list[set[str]]:
    groups: list[set[str]] = []
    for callee, pos_map, kw_map in call_args:
        if callee not in callee_groups:
            continue
        callee_params = callee_param_orders[callee]
        # Build mapping from callee param to caller param.
        callee_to_caller: dict[str, str] = {}
        for idx, pname in enumerate(callee_params):
            key = str(idx)
            if key in pos_map:
                callee_to_caller[pname] = pos_map[key]
        for kw, caller_name in kw_map.items():
            callee_to_caller[kw] = caller_name
        for group in callee_groups[callee]:
            mapped = {callee_to_caller.get(p) for p in group}
            mapped.discard(None)
            if len(mapped) > 1:
                groups.append(set(mapped))
    return groups


def analyze_file(path: Path, recursive: bool = True) -> dict[str, list[set[str]]]:
    tree = ast.parse(path.read_text())
    parent = ParentAnnotator()
    parent.visit(tree)
    parents = parent.parents

    funcs = _collect_functions(tree)
    fn_param_orders = {f.name: _param_names(f) for f in funcs}
    fn_use = {}
    fn_calls = {}
    for f in funcs:
        use_map, call_args = _analyze_function(f, parents)
        fn_use[f.name] = use_map
        fn_calls[f.name] = call_args

    groups_by_fn = {fn: _group_by_signature(use_map) for fn, use_map in fn_use.items()}

    if not recursive:
        return groups_by_fn

    changed = True
    while changed:
        changed = False
        for fn in fn_use:
            propagated = _propagate_groups(
                fn,
                fn_param_orders[fn],
                fn_calls[fn],
                groups_by_fn,
                fn_param_orders,
            )
            if not propagated:
                continue
            combined = _union_groups(groups_by_fn.get(fn, []) + propagated)
            if combined != groups_by_fn.get(fn, []):
                groups_by_fn[fn] = combined
                changed = True
    return groups_by_fn


def _iter_config_fields(path: Path) -> dict[str, set[str]]:
    """Best-effort extraction of @dataclass config bundles (fields ending with _fn)."""
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return {}
    bundles: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {getattr(d, "id", None) for d in node.decorator_list}
        if "dataclass" not in decorators and not node.name.endswith("Config"):
            continue
        fields: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                if name.endswith("_fn"):
                    fields.add(name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_fn"):
                        fields.add(target.id)
        if fields:
            bundles[node.name] = fields
    return bundles


_BUNDLE_MARKER = re.compile(r"dataflow-bundle:\s*(.*)")


def _iter_documented_bundles(path: Path) -> set[tuple[str, ...]]:
    """Return bundles documented via '# dataflow-bundle: a, b' markers."""
    bundles: set[tuple[str, ...]] = set()
    try:
        text = path.read_text()
    except Exception:
        return bundles
    for line in text.splitlines():
        match = _BUNDLE_MARKER.search(line)
        if not match:
            continue
        payload = match.group(1)
        if not payload:
            continue
        parts = [p.strip() for p in re.split(r"[,\s]+", payload) if p.strip()]
        if len(parts) < 2:
            continue
        bundles.add(tuple(sorted(parts)))
    return bundles


def _emit_dot(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> str:
    lines = [
        "digraph dataflow_grammar {",
        "  rankdir=LR;",
        "  node [fontsize=10];",
    ]
    for path, groups in groups_by_path.items():
        file_id = str(path).replace("/", "_").replace(".", "_")
        lines.append(f"  subgraph cluster_{file_id} {{")
        lines.append(f"    label=\"{path}\";")
        for fn, bundles in groups.items():
            if not bundles:
                continue
            fn_id = f"fn_{file_id}_{fn}"
            lines.append(f"    {fn_id} [shape=box,label=\"{fn}\"];")
            for idx, bundle in enumerate(bundles):
                bundle_id = f"b_{file_id}_{fn}_{idx}"
                label = ", ".join(sorted(bundle))
                lines.append(
                    f"    {bundle_id} [shape=ellipse,label=\"{label}\"];"
                )
                lines.append(f"    {fn_id} -> {bundle_id};")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _component_graph(groups_by_path: dict[Path, dict[str, list[set[str]]]]):
    nodes: dict[str, dict[str, str]] = {}
    adj: dict[str, set[str]] = defaultdict(set)
    bundle_map: dict[str, set[str]] = {}
    for path, groups in groups_by_path.items():
        file_id = str(path)
        for fn, bundles in groups.items():
            if not bundles:
                continue
            fn_id = f"fn::{file_id}::{fn}"
            nodes[fn_id] = {"kind": "fn", "label": f"{path.name}:{fn}"}
            for idx, bundle in enumerate(bundles):
                bundle_id = f"b::{file_id}::{fn}::{idx}"
                nodes[bundle_id] = {
                    "kind": "bundle",
                    "label": ", ".join(sorted(bundle)),
                }
                bundle_map[bundle_id] = bundle
                adj[fn_id].add(bundle_id)
                adj[bundle_id].add(fn_id)
    return nodes, adj, bundle_map


def _connected_components(nodes: dict[str, dict[str, str]], adj: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    comps: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        q: deque[str] = deque([node])
        seen.add(node)
        comp: list[str] = []
        while q:
            curr = q.popleft()
            comp.append(curr)
            for nxt in adj.get(curr, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(sorted(comp))
    return comps


def _render_mermaid_component(
    nodes: dict[str, dict[str, str]],
    bundle_map: dict[str, set[str]],
    adj: dict[str, set[str]],
    component: list[str],
    config_bundles_by_path: dict[Path, dict[str, set[str]]],
    documented_bundles_by_path: dict[Path, set[tuple[str, ...]]],
) -> tuple[str, str]:
    lines = ["```mermaid", "flowchart LR"]
    fn_nodes = [n for n in component if nodes[n]["kind"] == "fn"]
    bundle_nodes = [n for n in component if nodes[n]["kind"] == "bundle"]
    for n in fn_nodes:
        label = nodes[n]["label"].replace('"', "'")
        lines.append(f'  {abs(hash(n))}["{label}"]')
    for n in bundle_nodes:
        label = nodes[n]["label"].replace('"', "'")
        lines.append(f'  {abs(hash(n))}(({label}))')
    for n in component:
        for nxt in adj.get(n, ()):
            if nxt in component and nodes[n]["kind"] == "fn":
                lines.append(f"  {abs(hash(n))} --> {abs(hash(nxt))}")
    lines.append("  classDef fn fill:#cfe8ff,stroke:#2b6cb0,stroke-width:1px;")
    lines.append("  classDef bundle fill:#ffe9c6,stroke:#c05621,stroke-width:1px;")
    if fn_nodes:
        lines.append(
            "  class "
            + ",".join(str(abs(hash(n))) for n in fn_nodes)
            + " fn;"
        )
    if bundle_nodes:
        lines.append(
            "  class "
            + ",".join(str(abs(hash(n))) for n in bundle_nodes)
            + " bundle;"
        )
    lines.append("```")

    observed = [bundle_map[n] for n in bundle_nodes if n in bundle_map]
    bundle_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for bundle in observed:
        bundle_counts[tuple(sorted(bundle))] += 1
    component_paths: set[Path] = set()
    for n in fn_nodes:
        parts = n.split("::", 2)
        if len(parts) == 3:
            component_paths.add(Path(parts[1]))
    declared = set()
    documented = set()
    for path in component_paths:
        config_path = path.parent / "config.py"
        bundles = config_bundles_by_path.get(config_path)
        if bundles:
            for fields in bundles.values():
                declared.add(tuple(sorted(fields)))
        documented |= documented_bundles_by_path.get(path, set())
    observed_norm = {tuple(sorted(b)) for b in observed}
    observed_only = sorted(observed_norm - declared) if declared else sorted(observed_norm)
    declared_only = sorted(declared - observed_norm)
    documented_only = sorted(observed_norm & documented)
    def _tier(bundle: tuple[str, ...]) -> str:
        count = bundle_counts.get(bundle, 1)
        if declared:
            return "tier-1"
        if count > 1:
            return "tier-2"
        return "tier-3"
    summary_lines = [
        f"Functions: {len(fn_nodes)}",
        f"Observed bundles: {len(observed_norm)}",
    ]
    if not declared:
        summary_lines.append("Declared Config bundles: none found for this component.")
    if observed_only:
        summary_lines.append("Observed-only bundles (not declared in Configs):")
        for bundle in observed_only:
            tier = _tier(bundle)
            documented_flag = "documented" if bundle in documented else "undocumented"
            summary_lines.append(
                f"  - {', '.join(bundle)} ({tier}, {documented_flag})"
            )
    if documented_only:
        summary_lines.append("Documented bundles (dataflow-bundle markers):")
        summary_lines.extend(f"  - {', '.join(bundle)}" for bundle in documented_only)
    if declared_only:
        summary_lines.append("Declared Config bundles not observed in this component:")
        summary_lines.extend(f"  - {', '.join(bundle)}" for bundle in declared_only)
    summary = "\n".join(summary_lines)
    return "\n".join(lines), summary


def _emit_report(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
) -> tuple[str, list[str]]:
    nodes, adj, bundle_map = _component_graph(groups_by_path)
    components = _connected_components(nodes, adj)
    if groups_by_path:
        common = os.path.commonpath([str(p) for p in groups_by_path])
        root = Path(common)
    else:
        root = Path(".")
    config_bundles_by_path = {}
    documented_bundles_by_path = {}
    for path in sorted(root.rglob("config.py")):
        config_bundles_by_path[path] = _iter_config_fields(path)
    protocols = root / "prism_vm_core" / "protocols.py"
    if protocols.exists():
        config_bundles_by_path[protocols] = _iter_config_fields(protocols)
    for path in sorted(root.rglob("*.py")):
        documented_bundles_by_path[path] = _iter_documented_bundles(path)
    lines = [
        "<!-- dataflow-grammar -->",
        "Dataflow grammar audit (observed forwarding bundles).",
        "",
    ]
    if not components:
        return "\n".join(lines + ["No bundle components detected."]), []
    if len(components) > max_components:
        lines.append(
            f"Showing top {max_components} components of {len(components)}."
        )
    violations: list[str] = []
    for idx, comp in enumerate(components[:max_components], start=1):
        lines.append(f"### Component {idx}")
        mermaid, summary = _render_mermaid_component(
            nodes,
            bundle_map,
            adj,
            comp,
            config_bundles_by_path,
            documented_bundles_by_path,
        )
        lines.append(mermaid)
        lines.append("")
        lines.append("Summary:")
        lines.append("```")
        lines.append(summary)
        lines.append("```")
        lines.append("")
        for line in summary.splitlines():
            if "(tier-3, undocumented)" in line:
                violations.append(line.strip())
            if "(tier-1," in line or "(tier-2," in line:
                if "undocumented" in line:
                    violations.append(line.strip())
    if violations:
        lines.append("Violations:")
        lines.append("```")
        lines.extend(violations)
        lines.append("```")
    return "\n".join(lines), violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dot", default=None, help="Write DOT graph to file or '-' for stdout.")
    parser.add_argument("--report", default=None, help="Write Markdown report (mermaid) to file.")
    parser.add_argument("--max-components", type=int, default=10, help="Max components in report.")
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if undocumented/undeclared bundle violations are detected.",
    )
    args = parser.parse_args()
    paths = _iter_paths(args.paths)
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {}
    for path in paths:
        groups_by_path[path] = analyze_file(path, recursive=not args.no_recursive)
    if args.dot is not None:
        dot = _emit_dot(groups_by_path)
        if args.dot.strip() == "-":
            print(dot)
        else:
            Path(args.dot).write_text(dot)
        if args.report is None:
            return
    if args.report is not None:
        report, violations = _emit_report(groups_by_path, args.max_components)
        Path(args.report).write_text(report)
        if args.fail_on_violations and violations:
            raise SystemExit(
                f"dataflow grammar violations detected: {len(violations)}"
            )
        return
    for path, groups in groups_by_path.items():
        print(f"# {path}")
        for fn, bundles in groups.items():
            if not bundles:
                continue
            print(f"{fn}:")
            for bundle in bundles:
                print(f"  bundle: {sorted(bundle)}")
        print()


if __name__ == "__main__":
    main()
