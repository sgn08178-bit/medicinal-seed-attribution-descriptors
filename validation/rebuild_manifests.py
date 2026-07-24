#!/usr/bin/env python3
"""Rebuild the public repository SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".csv", ".md", ".py", ".txt", ".yaml", ".yml"}
TEXT_FILENAMES = {".gitignore"}


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
        return data.replace(b"\r\n", b"\n")
    return data


def sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def repository_manifest() -> None:
    output = REPOSITORY_ROOT / "MANIFEST.csv"
    rows = []
    for path in sorted(REPOSITORY_ROOT.rglob("*")):
        if (
            not path.is_file()
            or path == output
            or ".git" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        data = canonical_bytes(path)
        rows.append(
            {
                "relative_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "size_bytes": len(data),
                "sha256": sha256(data),
            }
        )
    write_csv(output, ["relative_path", "size_bytes", "sha256"], rows)


if __name__ == "__main__":
    repository_manifest()
    print("Rebuilt public repository manifest.")
