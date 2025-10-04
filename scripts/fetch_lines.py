#!/usr/bin/env python3
"""
Fetch sportsbook lines from either:
  - theoddsapi (official The Odds API v4)
  - oddsjam     (an "unofficial" JSON feed you provide)

Usage examples:
  # The Odds API:
  python scripts/fetch_lines.py \
      --provider theoddsapi \
      --api-key $ODDS_API_KEY \
      --markets player_pass_yds,player_rush_yds,player_receiving_yds,player_receptions,player_anytime_td \
      --out inputs/sportsbook_lines.csv

  # OddsJam (unofficial):
  python scripts/fetch_lines.py \
      --provider oddsjam \
      --url "$ODDSJAM_URL" \
      --out inputs/sportsbook_lines.csv

Outputs CSV with columns:
  player, market, line, over_odds, under_odds, yes_odds, no_odds, book, event_time
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests


# ------------------------------
# Markets we care about (standardized)
# ------------------------------
DEFAULT_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receiving_yds",
    "player_receptions",
    "player_anytime_td",
]

# ------------------------------
# Helpers
# ------------------------------
def _get(session: requests.Session, url: str, *, headers=None, params=None, tries=3, backoff=1.5) -> requests.Response:
    headers = headers or {}
    params = params or {}
    last_exc: Optional[Exception] = None
    for i in range(tries):
        try:
            r = session.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 429:
                # rate limit: back off
                sleep_for = backoff ** i
                time.sleep(sleep_for)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            sleep_for = backoff ** i
            time.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError("GET failed unexpectedly")

def _american_from_decimal(dec: Optional[float]) -> Optional[int]:
    """Convert decimal odds to American odds. Returns None if invalid."""
    try:
        if dec is None or pd.isna(dec) or float(dec) <= 1e-9:
            return None
        dec = float(dec)
        if dec >= 2.0:
            return int(round((dec - 1.0) * 100))
        else:
            return int(round(-100.0 / (dec - 1.0)))
    except Exception:
        return None

def _as_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _clean_string(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def _event_time_from_any(x: Any) -> Optional[str]:
    # Expecting ISO8601 strings; if not, try to pass-through
    if x is None:
        return None
    try:
        s = str(x).strip()
        return s if s else None
    except Exception:
        return None

def _normalize_markets(m: str) -> Optional[str]:
    """
    Normalize odds provider market labels to our internal labels.
    Add any alias you discover here.
    """
    if not m:
        return None
    s = str(m).lower().strip()

    # common aliases
    aliases = {
        "player_pass_yards": "player_pass_yds",
        "pass_yards": "player_pass_yds",
        "passing_yards": "player_pass_yds",

        "player_rush_yards": "player_rush_yds",
        "rush_yards": "player_rush_yds",
        "rushing_yards": "player_rush_yds",

        "player_receiving_yards": "player_receiving_yds",
        "receiving_yards": "player_receiving_yds",
        "rec_yards": "player_receiving_yds",

        "player_receptions": "player_receptions",
        "receptions": "player_receptions",

        "anytime_td": "player_anytime_td",
        "player_any_time_td": "player_anytime_td",
        "player_anytime_td": "player_anytime_td",
        "anytime_touchdown": "player_anytime_td",
    }
    if s in aliases:
        return aliases[s]
    return s  # maybe it's already standardized

# ------------------------------
# The Odds API (official) → normalized rows
# ------------------------------
def fetch_theoddsapi(
    api_key: str,
    markets: List[str],
    *,
    regions: str = "us",
    odds_format: str = "american",
    date_format: str = "iso",
) -> List[Dict[str, Any]]:
    """
    Calls: /v4/sports/americanfootball_nfl/odds
    normalizes into a list of rows (player, market, line, over_odds, under_odds, yes_odds, no_odds, book, event_time)
    """
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }

    rows: List[Dict[str, Any]] = []
    with requests.Session() as s:
        resp = _get(s, url, params=params)
        data = resp.json()

        # The Odds API returns a list of events, each with "bookmakers" and "markets"
        # For player markets, "outcomes" usually include a "name" (player) and optional "line".
        for event in data:
            event_time = _event_time_from_any(event.get("commence_time"))
            bookmakers = event.get("bookmakers", []) or []
            for bk in bookmakers:
                book = _clean_string(bk.get("title") or bk.get("key"))
                for mkt in bk.get("markets", []) or []:
                    market_key = _normalize_markets(mkt.get("key"))
                    if market_key not in markets:
                        continue
                    outcomes = mkt.get("outcomes", []) or []

                    # Two shapes:
                    # 1) Totals (Over/Under) for yardage or receptions: outcome names "Over"/"Under"
                    # 2) Anytime TD: often "Yes"/"No" with participant name possibly in outcome['description'] or in the event label.
                    #
                    # We try both shapes robustly.
                    over_line = None
                    under_line = None
                    over_odds = None
                    under_odds = None

                    for oc in outcomes:
                        name = _clean_string(oc.get("name"))
                        desc = _clean_string(oc.get("description"))
                        player = _clean_string(oc.get("player") or desc)  # TD markets might use 'description'
                        line = _as_float(oc.get("point") or oc.get("line"))
                        price = oc.get("price")
                        # price is American when oddsFormat=american
                        price = int(price) if price is not None and str(price).lstrip("-+").isdigit() else None

                        # Over/Under markets for yardage
                        if name and name.lower() == "over":
                            over_line = line
                            over_odds = price
                        elif name and name.lower() == "under":
                            under_line = line
                            under_odds = price

                        # Yes/No (Anytime TD)
                        yes_odds = None
                        no_odds = None
                        if name and name.lower() == "yes":
                            yes_odds = price
                            # try to find the player on the same outcome
                            if not player:
                                player = desc
                            rows.append({
                                "player": player,
                                "market": market_key,
                                "line": None,
                                "over_odds": None,
                                "under_odds": None,
                                "yes_odds": yes_odds,
                                "no_odds": None,
                                "book": book,
                                "event_time": event_time,
                            })
                            continue
                        if name and name.lower() == "no":
                            no_odds = price
                            if not player:
                                player = desc
                            rows.append({
                                "player": player,
                                "market": market_key,
                                "line": None,
                                "over_odds": None,
                                "under_odds": None,
                                "yes_odds": None,
                                "no_odds": no_odds,
                                "book": book,
                                "event_time": event_time,
                            })
                            continue

                    # If we saw both Over/Under, emit one row with the median line (or keep both).
                    if (over_line is not None or under_line is not None) and market_key != "player_anytime_td":
                        player_name = None
                        # The Odds API for player props often puts player name in outcome['description']
                        # Attempt to discover a player name out of outcomes if present:
                        for oc in outcomes:
                            if oc.get("description"):
                                player_name = oc.get("description")
                                break
                        rows.append({
                            "player": player_name,
                            "market": market_key,
                            "line": over_line if over_line is not None else under_line,
                            "over_odds": over_odds,
                            "under_odds": under_odds,
                            "yes_odds": None,
                            "no_odds": None,
                            "book": book,
                            "event_time": event_time,
                        })

    # Clean and drop empty players
    df = pd.DataFrame(rows)
    if not df.empty:
        # If player is missing, keep but label Unknown to avoid crashing downstream
        df["player"] = df["player"].fillna("Unknown")
    return df.to_dict(orient="records")

# ------------------------------
# OddsJam (unofficial) → normalized rows
# ------------------------------
def _flatten_oddsjam_json(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Heuristic flattener for OddsJam-style JSON. Because the feed is "unofficial",
    shapes vary. We try to recognize records that look like player props:
      - must have a recognizable market label
      - must carry either a line (for O/U) or a Yes/No price (for TD)
    This function is intentionally forgiving; adjust mapping as needed once you see the live feed.
    """
    # If the object is a list, recurse into each member.
    if isinstance(obj, list):
        for item in obj:
            yield from _flatten_oddsjam_json(item)
        return

    # If it's a dict, try to interpret it as a record, then recurse into values
    if isinstance(obj, dict):
        market = _normalize_markets(obj.get("market") or obj.get("marketKey") or obj.get("key") or obj.get("label"))
        player = _clean_string(obj.get("player") or obj.get("playerName") or obj.get("name") or obj.get("participant"))
        book = _clean_string(obj.get("book") or obj.get("bookmaker") or obj.get("source") or obj.get("sportsbook"))
        event_time = _event_time_from_any(obj.get("startTime") or obj.get("commence_time") or obj.get("eventTime"))

        # try common price/line paths in OddsJam exports (decimal odds common, sometimes american provided)
        line = _as_float(obj.get("line") or obj.get("point") or obj.get("total") or obj.get("handicap"))
        over_odds = _as_float(obj.get("overOdds") or obj.get("over_price") or obj.get("over"))
        under_odds = _as_float(obj.get("underOdds") or obj.get("under_price") or obj.get("under"))
        yes_odds = _as_float(obj.get("yesOdds") or obj.get("yes_price") or obj.get("yes"))
        no_odds = _as_float(obj.get("noOdds") or obj.get("no_price") or obj.get("no"))

        # If odds are decimal, convert to American.
        def conv_if_decimal(x):
            if x is None:
                return None
            # If it's already American (looks like +/- integers), keep it; else try decimal→American
            try:
                xi = int(x)
                # looks American already
                return xi
            except Exception:
                dec = _as_float(x)
                return _american_from_decimal(dec)

        over_odds = conv_if_decimal(over_odds)
        under_odds = conv_if_decimal(under_odds)
        yes_odds = conv_if_decimal(yes_odds)
        no_odds = conv_if_decimal(no_odds)

        # If it looks like a record, emit it.
        if market in DEFAULT_MARKETS and (line is not None or yes_odds is not None or no_odds is not None):
            yield {
                "player": player or "Unknown",
                "market": market,
                "line": line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "yes_odds": yes_odds,
                "no_odds": no_odds,
                "book": book,
                "event_time": event_time,
            }

        # Recurse into children for deeply-nested shapes
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from _flatten_oddsjam_json(v)

def fetch_oddsjam(url: str) -> List[Dict[str, Any]]:
    """
    Fetches an arbitrary JSON URL (your OddsJam endpoint) and tries to flatten it into rows.
    You may want to adjust `_flatten_oddsjam_json` once you see the actual shape.
    """
    with requests.Session() as s:
        resp = _get(s, url)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise RuntimeError("OddsJam URL did not return JSON")

    rows = list(_flatten_oddsjam_json(data))
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    # Normalize/clean
    for c in ["player", "market", "book"]:
        if c in df:
            df[c] = df[c].fillna("Unknown")
    return df.to_dict(orient="records")

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch sportsbook lines from a selected provider")
    ap.add_argument("--provider", choices=["theoddsapi", "oddsjam"], required=True,
                    help="Which provider to use")
    ap.add_argument("--api-key", default=os.getenv("ODDS_API_KEY"),
                    help="The Odds API key (only for provider=theoddsapi)")
    ap.add_argument("--url", default=os.getenv("ODDSJAM_URL"),
                    help="OddsJam JSON endpoint (only for provider=oddsjam)")
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS),
                    help="Comma-separated markets to request/keep (theoddsapi only)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.provider == "theoddsapi":
        if not args.api_key:
            print("[ERROR] --api-key (or ODDS_API_KEY env) is required for provider=theoddsapi", file=sys.stderr)
            sys.exit(2)
        markets = [m.strip() for m in args.markets.split(",") if m.strip()]
        rows = fetch_theoddsapi(args.api_key, markets=markets)
    elif args.provider == "oddsjam":
        if not args.url:
            print("[ERROR] --url (or ODDSJAM_URL env) is required for provider=oddsjam", file=sys.stderr)
            sys.exit(2)
        rows = fetch_oddsjam(args.url)
    else:
        print(f"[ERROR] Unknown provider: {args.provider}", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows, columns=[
        "player", "market", "line",
        "over_odds", "under_odds", "yes_odds", "no_odds",
        "book", "event_time"
    ])
    if df.empty:
        print("[WARN] No lines parsed; writing empty CSV", file=sys.stderr)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {len(df):,} rows -> {args.out}")

if __name__ == "__main__":
    main()
