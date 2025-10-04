#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/fetch_lines.py

Turn-key sportsbook line fetcher with pluggable providers:
  • OddsJam (unofficial JSON; schema can vary)
  • The Odds API (official, v4)

Design goals
------------
- One CLI for both providers (choose with --provider).
- Normalize output into a consistent, model-friendly CSV:
    player, market, line, over_odds, under_odds, yes_odds, no_odds,
    book, event_time, event_id, bookmaker_key, last_update
- Be robust to schema differences; don't crash on missing bits.
- Rate-limit / retry with exponential backoff.
- Allow auth headers for JSON feeds (e.g., bearer tokens).
- Optional: save raw JSON for debugging with --save-raw.
- Filters: markets, books, time windows, games only (no futures), etc.
- Always write a CSV (possibly empty) and exit 0 for CI stability.

Usage examples
--------------
  # OddsJam (unofficial JSON)
  python scripts/fetch_lines.py \
    --provider oddsjam \
    --url "$ODDSJAM_URL" \
    --out inputs/sportsbook_lines.csv \
    --save-raw raw/oddsjam.json \
    --markets player_pass_yds,player_rush_yds,player_receiving_yds,player_receptions,player_anytime_td \
    --books draftkings,fanduel,betmgm \
    --header "Authorization: Bearer <token>" \
    --header "X-Api-Key: <key>"

  # The Odds API (official)
  python scripts/fetch_lines.py \
    --provider theoddsapi \
    --api-key "$ODDS_API_KEY" \
    --out inputs/sportsbook_lines.csv \
    --markets player_pass_yds,player_rush_yds,player_receiving_yds,player_receptions,player_anytime_td \
    --books draftkings,fanduel,betmgm \
    --regions us \
    --odds-format american \
    --date-format iso

Notes
-----
- OddsJam is "unofficial": schema may change. This script is forgiving and
  will keep scanning nested JSON for recognizable records.
- The Odds API player props may require a paid plan; if 422 INVALID_MARKET,
  we log a warning and return empty rows (CSV still written).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests


# ------------------------------
# Constants & supported markets
# ------------------------------

# Internal normalized market keys:
SUPPORTED_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receiving_yds",
    "player_receptions",
    "player_anytime_td",
]

# Default set we care about if not specified
DEFAULT_MARKETS = ",".join(SUPPORTED_MARKETS)

# Output CSV schema (stable)
OUTPUT_COLUMNS = [
    "player", "market", "line",
    "over_odds", "under_odds", "yes_odds", "no_odds",
    "book", "event_time", "event_id", "bookmaker_key", "last_update",
]

THEODDSAPI_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


# ------------------------------
# Utilities
# ------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)

def err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

def mkdirp(path: str) -> None:
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def clean_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def american_from_decimal(dec: Optional[float]) -> Optional[int]:
    if dec is None or (isinstance(dec, float) and not math.isfinite(dec)):
        return None
    try:
        dec = float(dec)
    except Exception:
        return None
    if dec <= 1.0:
        return None
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))
    return int(round(-100.0 / (dec - 1.0)))

def parse_header_lines(header_lines: Sequence[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for line in header_lines:
        if not line:
            continue
        # Format: "Key: Value"
        m = re.match(r"^\s*([^:]+)\s*:\s*(.+)$", line)
        if not m:
            warn(f"Ignoring malformed header: {line!r}")
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        if k and v:
            headers[k] = v
    return headers

def event_time_from_any(x: Any) -> Optional[str]:
    # Leave as string; downstream can parse. Expect ISO strings or timestamps.
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def now_ts() -> int:
    return int(time.time())


# ------------------------------
# HTTP (retry/backoff)
# ------------------------------

def http_get(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    tries: int = 4,
    backoff: float = 1.8,
    timeout: int = 30,
) -> requests.Response:
    params = params or {}
    headers = headers or {}
    last_exc: Optional[Exception] = None
    for i in range(tries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            # Handle rate limit
            if r.status_code in (429, 503):
                sleep_for = backoff ** i
                warn(f"{r.status_code} from {url}; retrying in {sleep_for:.1f}s")
                time.sleep(sleep_for)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            sleep_for = backoff ** i
            warn(f"GET failed ({e}); retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError("GET failed unexpectedly without exception")


# ------------------------------
# Market normalization
# ------------------------------

ALIASES = {
    # passing yards
    "player_pass_yards": "player_pass_yds",
    "passing_yards": "player_pass_yds",
    "pass_yards": "player_pass_yds",
    # rushing yards
    "player_rush_yards": "player_rush_yds",
    "rushing_yards": "player_rush_yds",
    "rush_yards": "player_rush_yds",
    # receiving yards
    "player_receiving_yards": "player_receiving_yds",
    "receiving_yards": "player_receiving_yds",
    "rec_yards": "player_receiving_yds",
    # receptions
    "player_receptions": "player_receptions",
    "receptions": "player_receptions",
    # anytime td
    "anytime_td": "player_anytime_td",
    "player_any_time_td": "player_anytime_td",
    "player_anytime_td": "player_anytime_td",
    "anytime_touchdown": "player_anytime_td",
    "touchdown_anytime": "player_anytime_td",
}

def normalize_market(m: Any) -> Optional[str]:
    if m is None:
        return None
    s = str(m).strip().lower()
    if not s:
        return None
    return ALIASES.get(s, s)


# ------------------------------
# The Odds API path
# ------------------------------

def theoddsapi_invalid_market(status: int, payload: Any) -> bool:
    if status != 422:
        return False
    try:
        if isinstance(payload, dict):
            code = str(payload.get("error_code", "")).upper()
            return code == "INVALID_MARKET"
        # Some versions return a string with "Invalid markets"
        if isinstance(payload, str) and "invalid" in payload.lower():
            return True
    except Exception:
        pass
    return True

def fetch_theoddsapi(
    api_key: str,
    markets: List[str],
    regions: str,
    odds_format: str,
    date_format: str,
    bookmakers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Call The Odds API v4 and normalize results.
    We expect a list of events; each event has 'bookmakers' with 'markets'.
    """
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    rows: List[Dict[str, Any]] = []
    with requests.Session() as s:
        try:
            resp = http_get(s, THEODDSAPI_BASE, params=params)
            data = resp.json()
        except requests.HTTPError as e:
            # If invalid market, log and return empty (we still write CSV upstream)
            try:
                payload = e.response.json()
            except Exception:
                payload = e.response.text if hasattr(e, "response") else ""
            if theoddsapi_invalid_market(getattr(e, "response", resp).status_code, payload):
                warn("The Odds API: INVALID_MARKET (likely props unavailable on your plan).")
                return rows
            # other HTTP errors: log and return empty
            warn(f"The Odds API HTTP error: {e}")
            return rows
        except Exception as e:
            warn(f"The Odds API error: {e}")
            return rows

        if not isinstance(data, list):
            warn("Unexpected The Odds API payload type; expected list of events.")
            return rows

        for ev in data:
            event_id = ev.get("id")
            event_time = event_time_from_any(ev.get("commence_time"))
            # optional team fields
            # home_team = clean_str(ev.get("home_team"))
            # away_team = clean_str(ev.get("away_team"))

            for bm in ev.get("bookmakers", []) or []:
                book_title = clean_str(bm.get("title") or bm.get("key"))
                bookmaker_key = clean_str(bm.get("key"))
                bm_last = ev_last = clean_str(bm.get("last_update"))  # string if ISO

                mkts = bm.get("markets", []) or []
                for m in mkts:
                    mkey = normalize_market(m.get("key"))
                    if mkey not in markets:
                        continue
                    m_last = clean_str(m.get("last_update")) or bm_last

                    outcomes = m.get("outcomes", []) or []
                    # For player props, outcome 'description' often holds player name.
                    # Two main shapes:
                    #   • Over/Under with point + price
                    #   • Yes/No (TD) with price and desc=player
                    # We collect both shapes; combine where possible.

                    # Group by player description when present
                    # Over/Under: one row (we keep one "line" with both over/under odds)
                    # TD: two rows (yes/no) -> we combine into single row
                    temp_by_player: Dict[str, Dict[str, Any]] = {}

                    for oc in outcomes:
                        name = clean_str(oc.get("name"))  # "Over", "Under", "Yes", "No"
                        desc = clean_str(oc.get("description"))  # often player name
                        player = clean_str(oc.get("player")) or desc
                        if player is None:
                            # Sometimes not provided; keep "Unknown" to avoid dropping
                            player = "Unknown"

                        # Price in American if oddsFormat=american, else convert from decimal
                        price = oc.get("price")
                        if price is None:
                            # Sometimes provided as decimal
                            price = american_from_decimal(to_float(oc.get("decimal") or oc.get("odds")))
                        else:
                            try:
                                price = int(str(price).replace("+", ""))
                            except Exception:
                                price = None

                        point = to_float(oc.get("point") or oc.get("line"))

                        key = player
                        rec = temp_by_player.get(key, {
                            "player": player,
                            "market": mkey,
                            "line": None,
                            "over_odds": None,
                            "under_odds": None,
                            "yes_odds": None,
                            "no_odds": None,
                            "book": book_title,
                            "event_time": event_time,
                            "event_id": event_id,
                            "bookmaker_key": bookmaker_key,
                            "last_update": m_last or ev_last,
                        })

                        if name and name.lower() == "over":
                            rec["over_odds"] = price
                            if point is not None:
                                rec["line"] = point
                        elif name and name.lower() == "under":
                            rec["under_odds"] = price
                            if point is not None:
                                rec["line"] = point
                        elif name and name.lower() == "yes":
                            rec["yes_odds"] = price
                        elif name and name.lower() == "no":
                            rec["no_odds"] = price

                        temp_by_player[key] = rec

                    # Push merged rows for this market/book/event
                    for rec in temp_by_player.values():
                        rows.append(rec)

    return rows


# ------------------------------
# OddsJam path (unofficial JSON)
# ------------------------------

def find_iterables(obj: Any) -> Iterable[Any]:
    """Yield all dict-like/list-like descendants, including root."""
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from find_iterables(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from find_iterables(it)

def maybe_decimal_to_american(x: Any) -> Optional[int]:
    """
    Convert decimal odds to American when needed; pass through American ints/strings.
    """
    if x is None:
        return None
    s = str(x).strip()
    # Already American?
    try:
        return int(s)
    except Exception:
        pass
    # Try decimal
    return american_from_decimal(to_float(s))

@dataclass
class OJCandidate:
    player: Optional[str]
    market: Optional[str]
    line: Optional[float]
    over_odds: Optional[int]
    under_odds: Optional[int]
    yes_odds: Optional[int]
    no_odds: Optional[int]
    book: Optional[str]
    event_time: Optional[str]
    event_id: Optional[str]
    bookmaker_key: Optional[str]
    last_update: Optional[str]

def flatten_oddsjam(obj: Any, keep_markets: List[str], keep_books: Optional[List[str]]) -> List[Dict[str, Any]]:
    """
    Heuristic flattener for OddsJam JSON. We scan the tree for dicts that look like
    player prop records based on the presence of recognizable fields. Because the feed
    is unofficial and variable, we cast a wide net, then post-filter.
    """
    rows: List[Dict[str, Any]] = []

    for node in find_iterables(obj):
        if not isinstance(node, dict):
            continue

        # candidate extraction
        market_raw = node.get("market") or node.get("marketKey") or node.get("key") or node.get("label")
        market = normalize_market(market_raw)
        if not market:
            continue  # can't use it

        if market not in keep_markets:
            continue

        # player/participant
        player = clean_str(node.get("player") or node.get("playerName") or node.get("participant") or node.get("name"))

        # sportsbook/bookmaker/source
        book = clean_str(node.get("book") or node.get("bookmaker") or node.get("source") or node.get("sportsbook"))

        # event meta
        event_time = event_time_from_any(node.get("startTime") or node.get("commence_time") or node.get("eventTime"))
        event_id = clean_str(node.get("event_id") or node.get("eventId") or node.get("id") or node.get("game_id"))
        bookmaker_key = clean_str(node.get("bookmaker_key") or node.get("bookKey") or node.get("bookKey"))
        last_update = clean_str(node.get("last_update") or node.get("lastUpdate") or node.get("updated"))

        # lines and odds
        line = to_float(node.get("line") or node.get("point") or node.get("total") or node.get("handicap"))
        over_odds = maybe_decimal_to_american(node.get("overOdds") or node.get("over_price") or node.get("over"))
        under_odds = maybe_decimal_to_american(node.get("underOdds") or node.get("under_price") or node.get("under"))
        yes_odds = maybe_decimal_to_american(node.get("yesOdds") or node.get("yes_price") or node.get("yes"))
        no_odds = maybe_decimal_to_american(node.get("noOdds") or node.get("no_price") or node.get("no"))

        # If no odds at all, skip (nothing to price)
        if all(v is None for v in (over_odds, under_odds, yes_odds, no_odds)):
            continue

        # Book filter (if provided)
        if keep_books:
            if not book or book.lower() not in [b.lower() for b in keep_books]:
                continue

        rows.append({
            "player": player or "Unknown",
            "market": market,
            "line": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "yes_odds": yes_odds,
            "no_odds": no_odds,
            "book": book,
            "event_time": event_time,
            "event_id": event_id,
            "bookmaker_key": bookmaker_key,
            "last_update": last_update,
        })

    return rows

def fetch_oddsjam(urls: List[str], headers: Dict[str, str], keep_markets: List[str], keep_books: Optional[List[str]], save_raw: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_blobs: List[Any] = []

    with requests.Session() as s:
        for url in urls:
            try:
                resp = http_get(s, url, headers=headers)
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    warn(f"URL did not return JSON: {url}")
                    continue
                if save_raw:
                    raw_blobs.append(data)
                rows.extend(flatten_oddsjam(data, keep_markets, keep_books))
            except Exception as e:
                warn(f"OddsJam fetch error for {url}: {e}")

    # Save raw if requested
    if save_raw:
        try:
            mkdirp(save_raw)
            with open(save_raw, "w", encoding="utf-8") as f:
                json.dump(raw_blobs if len(urls) > 1 else (raw_blobs[0] if raw_blobs else {}), f, ensure_ascii=False)
            log(f"[info] saved raw JSON -> {save_raw}")
        except Exception as e:
            warn(f"Could not save raw JSON: {e}")

    return rows


# ------------------------------
# Post-processing / cleaning
# ------------------------------

def normalize_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # enforce columns & order
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[OUTPUT_COLUMNS]

    # Make sure types are sane
    for c in ["line"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["over_odds", "under_odds", "yes_odds", "no_odds"]:
        # odds should be integers; if floats, cast; keep NaN if missing
        df[c] = pd.to_numeric(df[c], errors="coerce").dropna().astype(int).reindex(df.index, fill_value=pd.NA)

    # Drop rows that are completely useless (no odds and no player)
    df["player"] = df["player"].fillna("Unknown")
    mask_any_odds = df[["over_odds", "under_odds", "yes_odds", "no_odds"]].notna().any(axis=1)
    df = df[mask_any_odds].copy()

    # Deduplicate exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def filter_by_books(df: pd.DataFrame, books: Optional[List[str]]) -> pd.DataFrame:
    if not books or df.empty or "book" not in df.columns:
        return df
    wanted = [b.lower().strip() for b in books]
    return df[df["book"].str.lower().isin(wanted)].reset_index(drop=True)


def filter_recent(df: pd.DataFrame, since_seconds: Optional[int]) -> pd.DataFrame:
    """
    Filter out entries whose 'last_update' is too old. We attempt to parse ISO strings
    and unix timestamps; if parsing fails, we keep the row.
    """
    if not since_seconds or df.empty:
        return df
    cutoff = now_ts() - since_seconds

    def to_unix(x: Any) -> Optional[int]:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        s = str(x).strip()
        if not s:
            return None
        # unix?
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
        # ISO-ish: try to strip non digits and parse a unix-ish fallback
        # (we keep it permissive; if we can't parse, we keep the row rather than drop)
        return None

    ts = df.get("last_update")
    if ts is None:
        return df
    keep_mask = []
    for v in ts:
        u = to_unix(v)
        if u is None:
            keep_mask.append(True)  # unknown; keep
        else:
            keep_mask.append(u >= cutoff)
    return df[pd.Series(keep_mask).values].reset_index(drop=True)


def keep_supported_markets(df: pd.DataFrame, markets: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    return df[df["market"].isin([m.lower() for m in markets])].reset_index(drop=True)


# ------------------------------
# CLI
# ------------------------------

def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch sportsbook lines from OddsJam or The Odds API, normalize, write CSV.")

    p.add_argument("--provider", required=True, choices=["oddsjam", "theoddsapi"],
                   help="which data source to use")

    # Shared
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--markets", default=DEFAULT_MARKETS,
                   help=f"comma-separated markets to keep (default: {DEFAULT_MARKETS})")
    p.add_argument("--books", default="",
                   help="comma-separated list of sportsbook names to keep (normalized lowercase compare)")
    p.add_argument("--since-seconds", type=int, default=0,
                   help="optional: keep only rows updated in the last N seconds (0=off)")
    p.add_argument("--save-raw", default="", help="optional: path to save raw JSON for debugging")

    # OddsJam specific
    p.add_argument("--url", default=os.getenv("ODDSJAM_URL", ""),
                   help="OddsJam JSON URL (can be a comma-separated list for multiple feeds)")
    p.add_argument("--header", action="append", default=[],
                   help='Extra headers for OddsJam, e.g. --header "Authorization: Bearer TOKEN" (repeatable)')

    # The Odds API specific
    p.add_argument("--api-key", default=os.getenv("ODDS_API_KEY", ""), help="The Odds API key")
    p.add_argument("--regions", default="us", help="regions param for The Odds API (default: us)")
    p.add_argument("--odds-format", default="american", help="american|decimal (default: american)")
    p.add_argument("--date-format", default="iso", help="iso|unix (default: iso)")
    p.add_argument("--bookmakers", default="", help="optional: comma-separated list for The Odds API 'bookmakers' param")

    return p.parse_args()


# ------------------------------
# Main
# ------------------------------

def main() -> int:
    args = build_cli()
    mkdirp(args.out)

    # Parse selections
    keep_markets = [m.strip().lower() for m in args.markets.split(",") if m.strip()]
    if not keep_markets:
        keep_markets = [m.lower() for m in SUPPORTED_MARKETS]

    books_list = [b.strip() for b in args.books.split(",") if b.strip()]
    bookmakers_list = [b.strip() for b in args.bookmakers.split(",") if b.strip()] if args.bookmakers else None

    rows: List[Dict[str, Any]] = []

    if args.provider == "oddsjam":
        if not args.url:
            err("OddsJam selected but --url (or ODDSJAM_URL env) is empty.")
            # Write empty CSV and exit 0 to keep CI alive
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.out, index=False)
            return 0

        headers = parse_header_lines(args.header or [])
        url_list = [u.strip() for u in args.url.split(",") if u.strip()]
        log(f"[info] OddsJam provider: {len(url_list)} URL(s), headers={list(headers.keys())}")
        rows = fetch_oddsjam(url_list, headers, keep_markets, books_list or None, save_raw=args.save_raw or None)

    elif args.provider == "theoddsapi":
        if not args.api_key:
            err("The Odds API selected but --api-key (or ODDS_API_KEY env) is empty.")
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.out, index=False)
            return 0

        log(f"[info] The Odds API provider; markets={keep_markets}, regions={args.regions}, bookmakers={bookmakers_list or 'all'}")
        rows = fetch_theoddsapi(
            api_key=args.api_key,
            markets=keep_markets,
            regions=args.regions,
            odds_format=args.odds_format,
            date_format=args.date_format,
            bookmakers=bookmakers_list,
        )
    else:
        err(f"Unknown provider: {args.provider}")
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.out, index=False)
        return 2

    # Normalize → DataFrame
    df = normalize_dataframe(rows)

    # Post filters
    df = keep_supported_markets(df, keep_markets)
    df = filter_by_books(df, books_list or None)
    df = filter_recent(df, args.since_seconds if args.since_seconds and args.since_seconds > 0 else None)

    # Final write
    df.to_csv(args.out, index=False)
    log(f"[OK] wrote {len(df):,} rows -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
