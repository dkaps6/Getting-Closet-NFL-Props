#!/usr/bin/env python3
"""
Robust sportsbook line fetcher for The Odds API (v4).

- Tries player markets first (if requested).
- On 422 INVALID_MARKET, falls back to core markets (h2h, spreads, totals).
- Writes CSV (even if empty) and exits 0 so the workflow continues.

Env:
  ODDS_API_KEY  (required to actually pull; if missing, we write an empty CSV)

Usage:
  python scripts/fetch_lines.py --out inputs/sportsbook_lines.csv
  # optional:
  --markets player_pass_yds,player_rush_yds,player_receiving_yds,player_receptions,player_anytime_td
  --regions us
  --bookmakers draftkings,fanduel,betmgm
  --odds-format american
  --date-format iso
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Tuple, Optional

import requests


ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
DEFAULT_PLAYER_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receiving_yds",
    "player_receptions",
    "player_anytime_td",
]
CORE_MARKETS = ["h2h", "spreads", "totals"]  # free/always-available on most plans

HEADERS = [
    "event_id",
    "sport_key",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update",
    "market_key",
    "market_last_update",
    "outcome_name",
    "outcome_price",
    "outcome_point",
    "outcome_description",
    "player_name",
]


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--markets", default=",".join(DEFAULT_PLAYER_MARKETS),
                   help="Comma-separated markets; script will auto-fallback on 422")
    p.add_argument("--regions", default="us")
    p.add_argument("--bookmakers", default="")
    p.add_argument("--odds-format", default="american")
    p.add_argument("--date-format", default="iso")
    # season/weeks accepted but not used by API; keep for interface compatibility
    p.add_argument("--season", default="")
    p.add_argument("--weeks", default="")
    return p.parse_args()


def odds_get(params: Dict[str, str]) -> Tuple[int, Any]:
    try:
        resp = requests.get(ODDS_API_BASE, params=params, timeout=30)
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return status, data
    except requests.RequestException as e:
        return 0, {"error": str(e)}


def is_invalid_market(status: int, data: Any) -> bool:
    if status != 422:
        return False
    # The Odds API returns {"message":"Invalid markets requested", "error_code":"INVALID_MARKET", ...}
    if isinstance(data, dict):
        code = str(data.get("error_code", "")).upper()
        if code == "INVALID_MARKET":
            return True
    # Fallback: treat any 422 as market invalid
    return True


def flatten_rows(events: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for ev in events:
        event_id = ev.get("id")
        sport_key = ev.get("sport_key")
        commence_time = ev.get("commence_time")
        home_team = ev.get("home_team")
        away_team = ev.get("away_team")

        for bm in ev.get("bookmakers", []):
            bm_key = bm.get("key")
            bm_title = bm.get("title")
            bm_last = bm.get("last_update")

            for mk in bm.get("markets", []):
                m_key = mk.get("key")
                m_last = mk.get("last_update")

                for oc in mk.get("outcomes", []):
                    name = oc.get("name")
                    price = oc.get("price")
                    point = oc.get("point")
                    desc = oc.get("description")  # some props include description
                    # some providers include "player" or "participant" field
                    player = oc.get("player") or oc.get("participant")

                    rows.append([
                        event_id,
                        sport_key,
                        commence_time,
                        home_team,
                        away_team,
                        bm_key,
                        bm_title,
                        bm_last,
                        m_key,
                        m_last,
                        name,
                        price,
                        point,
                        desc,
                        player,
                    ])
    return rows


def write_csv(path: str, rows: List[List[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        w.writerows(rows)


def fetch_with_markets(api_key: str, markets: List[str], regions: str,
                       bookmakers: str, odds_fmt: str, date_fmt: str) -> Tuple[int, Any]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "oddsFormat": odds_fmt,
        "dateFormat": date_fmt,
        "markets": ",".join(markets),
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    return odds_get(params)


def main() -> int:
    args = build_args()
    api_key = os.environ.get("ODDS_API_KEY", "").strip()

    if not api_key:
        print("[info] No ODDS_API_KEY found; writing empty CSV and continuing.")
        write_csv(args.out, [])
        return 0

    requested_markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    # First try requested (likely player props)
    print(f"[info] Requesting markets: {requested_markets}")
    status, data = fetch_with_markets(
        api_key, requested_markets, args.regions, args.bookmakers, args.odds_format, args.date_format
    )

    if is_invalid_market(status, data):
        print("[warn] The Odds API returned 422 INVALID_MARKET for requested markets "
              f"{requested_markets}. This often means your plan does not include player props.")
        # Fallback to core markets
        print(f"[info] Falling back to core markets: {CORE_MARKETS}")
        status, data = fetch_with_markets(
            api_key, CORE_MARKETS, args.regions, args.bookmakers, args.odds_format, args.date_format
        )

    if status == 200 and isinstance(data, list):
        rows = flatten_rows(data)
        print(f"[info] Received {len(rows)} flattened rows from Odds API.")
        write_csv(args.out, rows)
        return 0

    # Any other non-200: don’t fail the pipeline – log and write empty CSV
    print(f"[warn] Odds API non-success status {status}. "
          f"Writing empty CSV. Payload preview: {str(data)[:300]}")
    write_csv(args.out, [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
