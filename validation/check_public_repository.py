#!/usr/bin/env python3
"""Dependency-free validation for the public repository."""

from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.csv"
TEXT_SUFFIXES = {".md", ".py", ".txt", ".yaml", ".yml"}
FORBIDDEN_FRAGMENTS = {
    "ScientificReports_submission",
    "manuscript_v3",
    "stage1_model_performance_comparison_runs",
    "stage2_attribution_maps/runs",
    "stage3_descriptor_association/runs",
    "stage5_descriptor_context_validation",
    "FOR_GPT",
    "GPT-ready",
    "run_stage7c_all_valid_descriptor_classifier.py",
    "revise_stage7c_for_manuscript.py",
    "create_submission_ready_supplementary_tables.py",
    "postprocess_png_v3.py",
    "C:\\Users\\",
    "/home/",
    "/mnt/",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_files() -> list[Path]:
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path != MANIFEST
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    ]


def validate_manifest(files: list[Path]) -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected = {path.relative_to(ROOT).as_posix(): path for path in files}
    recorded = {row["relative_path"]: row for row in rows}
    if expected.keys() != recorded.keys():
        missing = sorted(expected.keys() - recorded.keys())
        stale = sorted(recorded.keys() - expected.keys())
        raise ValueError(f"Manifest file-set mismatch; missing={missing}, stale={stale}")

    for relative_path, path in expected.items():
        row = recorded[relative_path]
        if int(row["size_bytes"]) != path.stat().st_size:
            raise ValueError(f"Manifest size mismatch: {relative_path}")
        if row["sha256"] != sha256(path):
            raise ValueError(f"Manifest SHA-256 mismatch: {relative_path}")

    print(f"Manifest: PASS ({len(rows)} files)")


def validate_python(files: list[Path]) -> None:
    python_files = [path for path in files if path.suffix == ".py"]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print(f"Python syntax: PASS ({len(python_files)} files)")


def validate_public_paths(files: list[Path]) -> None:
    hits: list[str] = []
    for path in files:
        if path == Path(__file__).resolve() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for fragment in FORBIDDEN_FRAGMENTS:
            if fragment in text:
                hits.append(f"{path.relative_to(ROOT).as_posix()}: {fragment}")
    if hits:
        raise ValueError("Forbidden private/stale path fragments:\n" + "\n".join(hits))
    print("Private/stale path scan: PASS")


def main() -> None:
    files = repository_files()
    validate_manifest(files)
    validate_python(files)
    validate_public_paths(files)
    print("Public repository validation: PASS")


if __name__ == "__main__":
    main()
