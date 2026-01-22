#!/usr/bin/env python3
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import analyze

TOKEN_PATTERN = analyze.TOKEN_PATTERN


def _tokenize(text, stopwords, pattern=TOKEN_PATTERN):
    token_re = re.compile(pattern)
    tokens = [t.lower() for t in token_re.findall(text)]
    return [t for t in tokens if t not in stopwords]


def _bigrams(tokens):
    return list(zip(tokens, tokens[1:]))


def _stats_for(path, stopwords):
    text = Path(path).read_text()
    tokens = _tokenize(text, stopwords)
    token_counts = Counter(tokens)
    token_set = set(token_counts)
    bigram_list = _bigrams(tokens)
    bigram_counts = Counter(bigram_list)
    bigram_set = set(bigram_counts)
    return {
        "token_counts": token_counts,
        "token_set": token_set,
        "bigram_counts": bigram_counts,
        "bigram_set": bigram_set,
    }


def _top_items(items, counts_a, counts_b=None, limit=12):
    counts_b = counts_b or Counter()
    scored = []
    for item in items:
        scored.append((counts_a[item] + counts_b[item], item))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [item for _, item in scored[:limit]]


def _top_items_single(items, counts, limit=12):
    scored = []
    for item in items:
        scored.append((counts[item], item))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [item for _, item in scored[:limit]]


def _fmt_token_list(items):
    return ", ".join(items)


def _fmt_bigram_list(items):
    return ", ".join([f"{a} {b}" for a, b in items])


def _doc_sort_key(path):
    match = re.search(r"in-(\d+)\.md$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def _section_for(doc_path, prior_path, compare_paths, stopwords):
    cur = _stats_for(doc_path, stopwords)
    out = []
    out.append(f"## {doc_path.as_posix()}")
    out.append(f"- Unique tokens: {len(cur['token_set'])}")
    out.append(f"- Unique bigrams: {len(cur['bigram_set'])}")

    if prior_path is None:
        out.append("- Prior version: none")
    else:
        prior = _stats_for(prior_path, stopwords)
        inter = cur["token_set"] & prior["token_set"]
        sym_diff = cur["token_set"] ^ prior["token_set"]
        only_cur = cur["token_set"] - prior["token_set"]
        only_prior = prior["token_set"] - cur["token_set"]
        top_inter = _top_items(inter, cur["token_counts"], prior["token_counts"])
        top_only_cur = _top_items_single(only_cur, cur["token_counts"])
        top_only_prior = _top_items_single(only_prior, prior["token_counts"])
        bigram_inter = cur["bigram_set"] & prior["bigram_set"]
        top_bigram = _top_items(
            bigram_inter, cur["bigram_counts"], prior["bigram_counts"]
        )
        out.append(f"- Prior version: {prior_path.as_posix()}")
        out.append(f"  - Intersection: {len(inter)} | top: {_fmt_token_list(top_inter)}")
        out.append(
            f"  - Symmetric difference: {len(sym_diff)} (only in {doc_path.as_posix()}: {len(only_cur)}, only in {prior_path.as_posix()}: {len(only_prior)})"
        )
        out.append(
            f"    - Only in {doc_path.as_posix()}: {_fmt_token_list(top_only_cur)}"
        )
        out.append(
            f"    - Only in {prior_path.as_posix()}: {_fmt_token_list(top_only_prior)}"
        )
        out.append(
            f"  - Wedge product (bigram intersection): {len(bigram_inter)} | top: {_fmt_bigram_list(top_bigram)}"
        )

    for label, path in compare_paths:
        other = _stats_for(path, stopwords)
        inter = cur["token_set"] & other["token_set"]
        sym_diff = cur["token_set"] ^ other["token_set"]
        only_cur = cur["token_set"] - other["token_set"]
        only_other = other["token_set"] - cur["token_set"]
        top_inter = _top_items(inter, cur["token_counts"], other["token_counts"])
        top_only_cur = _top_items_single(only_cur, cur["token_counts"])
        top_only_other = _top_items_single(only_other, other["token_counts"])
        bigram_inter = cur["bigram_set"] & other["bigram_set"]
        top_bigram = _top_items(
            bigram_inter, cur["bigram_counts"], other["bigram_counts"]
        )
        out.append(f"- Compare: {label}")
        out.append(f"  - Intersection: {len(inter)} | top: {_fmt_token_list(top_inter)}")
        out.append(
            f"  - Symmetric difference: {len(sym_diff)} (only in {doc_path.as_posix()}: {len(only_cur)}, only in {label}: {len(only_other)})"
        )
        out.append(
            f"    - Only in {doc_path.as_posix()}: {_fmt_token_list(top_only_cur)}"
        )
        out.append(f"    - Only in {label}: {_fmt_token_list(top_only_other)}")
        out.append(
            f"  - Wedge product (bigram intersection): {len(bigram_inter)} | top: {_fmt_bigram_list(top_bigram)}"
        )

    return "\n".join(out)


def _build_report(docs_dir, compare_paths, stopwords):
    docs = sorted(Path(docs_dir).glob("in-*.md"), key=_doc_sort_key)
    glossary = Path(docs_dir) / "glossary.md"
    has_glossary = glossary.exists()
    title = "# Audit: in/in-*.md + in/glossary.md" if has_glossary else "# Audit: in/in-*.md"
    header = [
        title,
        "",
        "Generated by `scripts/audit_in_versions.py`. Do not edit by hand.",
        "",
        "Methodology:",
        "- Tokenization: `[^\\W\\d_][\\w]*` (unicode letters, then word chars), lowercased.",
        "- Stopwords: English common words + Python keywords (see script).",
        "- Sets are unique tokens after filtering.",
        "- Wedge product: intersection of adjacent token bigram sets (ordered pairs).",
    ]
    if has_glossary:
        header.append("- Glossary: `in/glossary.md` is included as a standalone appendix.")
    header.append("")

    sections = []
    prior = None
    for doc in docs:
        sections.append(_section_for(doc, prior, compare_paths, stopwords))
        prior = doc
    if has_glossary:
        sections.append(_section_for(glossary, None, compare_paths, stopwords))
    return "\n".join(header + sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate audit_in_versions.md")
    parser.add_argument("--docs-dir", default="in")
    parser.add_argument("--output", default="audit_in_versions.md")
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument(
        "--compare",
        nargs="*",
        default=["prism_vm.py", "IMPLEMENTATION_PLAN.md"],
        help="Additional comparison paths (default: prism_vm.py IMPLEMENTATION_PLAN.md)",
    )
    args = parser.parse_args()

    compare_paths = [(Path(p).as_posix(), Path(p)) for p in args.compare]
    stopwords = analyze.load_stopwords(include_python=True)
    report = _build_report(args.docs_dir, compare_paths, stopwords)
    if args.stdout:
        print(report, end="")
        return
    Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
