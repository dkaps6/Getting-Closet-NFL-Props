#!/usr/bin/env python3
"""
fetch_features.py
Robust feature fetcher (nflverse + NWS) with graceful fallbacks.

- Tries to import and use nfl_data_py.
- If the package is missing or any pull fails, writes an empty CSV with headers
  so downstream steps can continue.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import List, Any

# -------------------------
# Minimal schema expected downstream (adjust as your engine needs)
# -------------------------
FEATURE_HEADERS: List[str] = [
    "game_id",
    "week",
    "season",
    "home_team",
    "away_team",
    "pace_sec_play",
    "pass_rate",
    "rush_rate",
    "temp_f",
    "wind_mph",
    "precip_mm",
    "roof",
    "surface",
]

def write_csv(path: str, rows: List[List[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FEATURE_HEADERS)
        w.writerows(rows)

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output CSV for features")
    p.add_argument("--season", default="", help="Optional season (e.g., 2025)")
    p.add_argument("--week", default="", help="Optional single week (e.g., 6)")
    return p.parse_args()

def fetch_using_nfl_data_py(season: str, week: str) -> List[List[Any]]:
    """Pull a small set of features with nfl_data_py.
       Keep this conservative; augment later as needed.
    """
    import nfl_data_py as nfl  # type: ignore

    # Example: schedule gives us game_id/home/away/time; team_stats/efficiency can be merged here
    schedule = nfl.import_schedules([int(season)]) if season else nfl.import_schedules(years=[ ])
    if week:
        try:
            week_i = int(week)
            schedule = schedule[schedule["week"] == week_i]
        except Exception:
            pass

    rows: List[List[Any]] = []
    # Use defaults/NA for now; wire in real env/pace once available
    for _, g in schedule.iterrows():
        game_id = g.get("game_id") or g.get("game_key") or ""
        wk = g.get("week")
        yr = g.get("season")
        home = g.get("home_team")
        away = g.get("away_team")

        # place-holders; replace with real merges (pace, pass rate, weather) as needed
        pace = None
        pass_rate = None
        rush_rate = None
        temp_f = None
        wind_mph = None
        precip_mm = None
        roof = None
        surface = None

        rows.append([
            game_id, wk, yr, home, away, pace, pass_rate, rush_rate,
            temp_f, wind_mph, precip_mm, roof, surface
        ])

    return rows

def main() -> int:
    args = build_args()

    # Try import and fetch
    try:
        rows = fetch_using_nfl_data_py(args.season, args.week)
        print(f"[info] nfl_data_py fetch ok: {len(rows)} rows")
        write_csv(args.out, rows)
        return 0
    except ModuleNotFoundError as e:
        print(f"[warn] {e}. nfl_data_py not installed. Writing empty features and continuing.")
    except Exception as e:
        print(f"[warn] Feature fetch failed: {e}. Writing empty features and continuing.")

    # Fallback – write empty with headers so pipeline proceeds
    write_csv(args.out, [])
    return 0

if __name__ == "__main__":
    sys.exit(main())
