"""Canonical manual-orientation correction manifest utilities.

The manifest records 24 unique images. ``initial_operation`` reproduces the
manual QC correction count reported in the manuscript (15 horizontal flips,
4 vertical flips, and 5 rotations). ``postprocess_operation`` records the
additional final-stage adjustment for four of those same images; it does not
increase the number of uniquely corrected images.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path


REPOSITORY_ROOT = Path(
    os.environ.get("MEDICINAL_SEED_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
DATA_ROOT = Path(
    os.environ.get("MEDICINAL_SEED_DATA_ROOT", REPOSITORY_ROOT / "data")
).resolve()
DEFAULT_CORRECTION_CSV = Path(
    os.environ.get(
        "MEDICINAL_SEED_CORRECTION_CSV",
        DATA_ROOT / "metadata" / "manual_orientation_corrections.csv",
    )
).resolve()

VALID_INITIAL_OPERATIONS = {
    "none", "hflip", "vflip", "rot90cw", "rot90ccw", "rot135cw", "rot225cw"
}
VALID_POSTPROCESS_OPERATIONS = {"none", "hflip", "rot90cw"}


def read_correction_manifest(csv_path: str | Path = DEFAULT_CORRECTION_CSV) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required manual-correction manifest not found: {path}. "
            "Preprocessing is stopped to prevent silent omission of the 24 corrections."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]

    required = {"stem", "initial_operation", "postprocess_operation"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Correction manifest must contain columns: {sorted(required)}")

    stems = [row["stem"] for row in rows]
    if len(rows) != 24 or len(set(stems)) != 24:
        raise ValueError("Correction manifest must contain exactly 24 unique image stems")

    initial = [row["initial_operation"].lower() for row in rows]
    post = [row["postprocess_operation"].lower() for row in rows]
    if not set(initial).issubset(VALID_INITIAL_OPERATIONS):
        raise ValueError(f"Unsupported initial operation(s): {sorted(set(initial) - VALID_INITIAL_OPERATIONS)}")
    if not set(post).issubset(VALID_POSTPROCESS_OPERATIONS):
        raise ValueError(f"Unsupported postprocess operation(s): {sorted(set(post) - VALID_POSTPROCESS_OPERATIONS)}")

    counts = {
        "hflip": initial.count("hflip"),
        "vflip": initial.count("vflip"),
        "rotation": sum(operation.startswith("rot") for operation in initial),
    }
    if counts != {"hflip": 15, "vflip": 4, "rotation": 5}:
        raise ValueError(f"Correction counts do not match the manuscript: {counts}")
    return rows


def load_operations(
    column: str,
    csv_path: str | Path = DEFAULT_CORRECTION_CSV,
    include_none: bool = False,
) -> dict[str, str]:
    if column not in {"initial_operation", "postprocess_operation"}:
        raise ValueError(f"Unsupported correction column: {column}")
    operations = {row["stem"]: row[column].lower() for row in read_correction_manifest(csv_path)}
    if not include_none:
        operations = {stem: operation for stem, operation in operations.items() if operation != "none"}
    return operations
