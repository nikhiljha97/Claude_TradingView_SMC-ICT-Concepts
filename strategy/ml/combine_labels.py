#!/usr/bin/env python3
"""Combine multiple labeled CSVs into one training file."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default=str(PROCESSED_DIR / "combined_labels.csv"))
    args = parser.parse_args()

    fieldnames = None
    rows = []
    for path in args.inputs:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    print({"rows": len(rows), "inputs": len(args.inputs)})


if __name__ == "__main__":
    main()
