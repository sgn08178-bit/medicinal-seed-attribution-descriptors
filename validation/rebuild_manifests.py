#!/usr/bin/env python3
"""Rebuild the public repository SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        rows.append(
            {
                "relative_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(output, ["relative_path", "size_bytes", "sha256"], rows)


if __name__ == "__main__":
    repository_manifest()
    print("Rebuilt public repository manifest.")
