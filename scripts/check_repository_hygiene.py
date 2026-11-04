#!/usr/bin/env python3
"""Reject local AI output in the Git index, including files added with -f."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import PurePosixPath


LOCAL_ROOTS = {".codex", ".opencode", ".tools", "report", "reports", "results", "models", "runs", "logs"}
MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}
NOTICES = {"LICENSE.md", "NOTICE.md", "THIRD_PARTY_NOTICES.md"}
RUN_DOCUMENT = re.compile(
    r"(?:^|_)(?:REPORT|REVIEW|AUDIT|STATUS|RECEIPT|RESULT|DRAFT|CANDIDATE|PILOT|LAUNCH|RUNNING)(?:_|$)",
    re.IGNORECASE,
)
GENERATED_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".log", ".bundle", ".patch", ".zip", ".tar", ".tgz"}


def violation(filename: str) -> str | None:
    path = PurePosixPath(filename)
    if path.parts[0] in LOCAL_ROOTS or path.parts[0].startswith("_cleanup_"):
        return "local output belongs in .codex/ or .opencode/"
    if path.parts[0] == "model" and len(path.parts) > 1 and re.search(r"_u\d", path.parts[1]):
        return "exported policy packages belong in the local artifacts/ directory"
    if path.suffix.lower() in GENERATED_SUFFIXES or filename.endswith(".tar.gz"):
        return "generated checkpoints, logs and archives must stay local"
    if path.suffix.lower() in MARKDOWN_SUFFIXES:
        if filename in {"AGENTS.md", "README.md"} or path.name in NOTICES:
            return None
        if path.name == "README.md" and path.parts[0] in {"assets", "model", "tools"}:
            return None
        if filename == "assets/urdf_v40/evidence/V40_TRAINING_DESIGN.md":
            return None
        if path.parts[0] != "docs":
            return "maintained documentation belongs in docs/; working notes stay local"
        if RUN_DOCUMENT.search(path.stem):
            return "run reports, reviews and working drafts belong in a local report/ directory"
    if path.parts[0] == "docs" and (
        any(part in {"evidence", "report", "reports"} for part in path.parts[1:-1])
        or path.name in {"runC_curves.png", "training_curves.png"}
    ):
        return "generated evidence does not belong in docs/"
    return None


def main() -> int:
    if len(sys.argv) != 1:
        print("usage: python scripts/check_repository_hygiene.py", file=sys.stderr)
        return 2
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    failures = [(name, violation(name)) for name in names if name]
    failures = [(name, reason) for name, reason in failures if reason]
    for name, reason in failures:
        print(f"{name}: {reason}", file=sys.stderr)
    if failures:
        print("Keep AI reports in .codex/report/ or .opencode/report/ and unstage them.", file=sys.stderr)
        return 1
    print("Repository hygiene: no local AI output is tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
