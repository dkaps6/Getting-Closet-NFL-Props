#!/usr/bin/env python3
"""
Bootstrap-safe engine.
- If inputs/straights.csv exists, loads it, otherwise logs and exits 0.
- Writes a tiny outputs/heartbeat.csv so artifact step has something.
"""

from pathlib import Path
import pandas as pd
from datetime import datetime

def main():
    inputs = Path("inputs")
    outputs = Path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    straight_path = inputs / "straights.csv"
    if straight_path.exists():
        try:
            df = pd.read_csv(str(straight_path))
            print(f"[engine] straights.csv rows={len(df)}, cols={len(df.columns)}")
        except Exception as e:
            print(f"[engine] WARN: could not read inputs/straights.csv: {e}")
    else:
        print("[engine] inputs/straights.csv not present (ok while bootstrapping).")

    # Always write a tiny file so we can prove outputs worked
    hb = outputs / "heartbeat.csv"
    hb.write_text(f"timestamp,ok\n{datetime.utcnow().isoformat()}Z,1\n")
    print(f"[engine] wrote {hb}")

if __name__ == "__main__":
    main()
