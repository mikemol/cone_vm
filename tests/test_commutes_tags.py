import re
from pathlib import Path


COMMUTES_RE = re.compile(r"# COMMUTES: .*\\[test: ([^\\]]+)\\]")


def test_commutes_tags_reference_existing_tests():
    repo_root = Path(__file__).resolve().parents[1]
    tags = []
    for path in repo_root.rglob("*.py"):
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in COMMUTES_RE.finditer(text):
            tags.append((path, match.group(1)))

    assert tags, "COMMUTES tags missing"

    for source, node in tags:
        file_part, sep, test_name = node.partition("::")
        assert sep, f"COMMUTES tag missing pytest node in {source}"
        test_path = repo_root / file_part
        assert test_path.exists(), f"COMMUTES tag references missing file {node}"
        content = test_path.read_text(encoding="utf-8")
        pattern = rf"def {re.escape(test_name)}\\b"
        assert re.search(pattern, content), f"COMMUTES tag references missing test {node}"
