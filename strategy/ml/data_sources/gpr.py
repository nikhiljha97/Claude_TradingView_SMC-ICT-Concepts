#!/usr/bin/env python3
"""Download the official daily AI-GPR geopolitical risk dataset."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.ml.common import RAW_DIR, ensure_dirs


AI_GPR_DAILY_URL = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RAW_DIR / "gpr" / "ai_gpr_data_daily.csv"))
    args = parser.parse_args()

    ensure_dirs()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(AI_GPR_DAILY_URL, timeout=60) as response:
        out.write_bytes(response.read())
    print(out)


if __name__ == "__main__":
    main()
