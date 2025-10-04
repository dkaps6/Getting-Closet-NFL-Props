#!/usr/bin/env python3
"""
Bootstrap-safe advanced step.
- Imports pandas/numpy to prove deps are installed.
- If inputs/sportsbook_lines.csv exists, reads it and prints a short summary.
- --dry-run flag avoids any side-effects.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print("NumPy:", np.__version__)
    print("pandas:", pd.__version__)
    print("Python:", sys.version.split()[0])

    lines_path = Path("inputs/sportsbook_lines.csv")
    if lines_path.exists():
        try:
            df = pd.read_csv(lines_path)
            print(f"[advanced] sportsbook_lines.csv rows={len(df)}, cols={len(df.columns)}")
            print("[advanced] head(3):")
            print(df.head(3).to_string(index=False))
        except Exception as e:
            print(f"[advanced] WARN: could not read inputs/sportsbook_lines.csv: {e}")
    else:
        print("[advanced] inputs/sportsbook_lines.csv not present (ok while bootstrapping).")

    if args.dry_run:
        print("[advanced] dry run complete.")
        return 0

    # Place your real advanced features here
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
