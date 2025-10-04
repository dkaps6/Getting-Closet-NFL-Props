# scripts/fetch_lines.py
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict, Any
import requests
import pandas as pd


SPORT = "americanfootball_nfl"
BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"

# Try markets individually; unsupported ones are skipped without failing the job.
MARKETS: List[str] = [
    # yards + receptions
    "player_pass_yds",
    "player_rush_yds",
    "player_receiving_yds",
    "player_receptions",
    # touchdowns (these may be plan-dependent; harmlessly skipped if unsupported)
    "player_pass_tds",
    "player_rush_tds",
    "player_receiving_tds",
    "player_anytime_td",
]

DEFAULT_PARAMS = {
    "regions": "us",
    "oddsFormat": "american",
    "dateFormat": "iso",
    # You can restrict by bookmakers if you want:
    # "bookmakers": "draftkings,fanduel,betmgm,caesars",
}


def _get(api_key: str, market: str) -> List[Dict[str, Any]]:
    params = dict(DEFAULT_PARAMS)
    params["apiKey"] = api_key
    params["markets"] = market

    try:
        r = requests.get(BASE_URL, params=params, timeout=25)
    except requests.RequestException as e:
        print(f"[WARN] Network error for market '{market}': {e}", file=sys.stderr)
        return []

    if r.status_code == 200:
        try:
            return r.json()
        except Exception as e:
            print(f"[WARN] JSON decode error for '{market}': {e}", file=sys.stderr)
            return []

    # Common: 422 Unprocessable Entity when a market isn’t available on your plan/region
    print(f"[WARN] Market '{market}' returned {r.status_code}: {r.text[:180]} ...", file=sys.stderr)
    return []


def _flatten(games: List[Dict[str, Any]], market_key: str) -> pd.DataFrame:
    """
    Flatten The Odds API JSON to a standard rows table:
      player, market, line, odds, book, game_id, commence_time, home_team, away_team
    """
    rows = []
    for g in games:
        gid = g.get("id")
        commence = g.get("commence_time")
        home = g.get("home_team")
        away = g.get("away_team")
        bms = g.get("bookmakers", []) or []

        for bm in bms:
            book = bm.get("title") or bm.get("key")
            for m in bm.get("markets", []):
                if m.get("key") != market_key:
                    continue
                for o in m.get("outcomes", []):
                    # For player props, 'name' is typically the player's name.
                    player = o.get("name")
                    odds = o.get("price")
                    line = o.get("point")
                    # normalize row
                    rows.append({
                        "player": player,
                        "market": market_key,
                        "line": line,
                        "odds": odds,
                        "book": book,
                        "game_id": gid,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "player", "market", "line", "odds", "book", "game_id",
            "commence_time", "home_team", "away_team"
        ])
    df = pd.DataFrame(rows)
    # drop clearly incomplete rows
    df = df.dropna(subset=["player", "odds"], how="any")
    return df


def fetch_all(api_key: str) -> pd.DataFrame:
    frames = []
    for mk in MARKETS:
        print(f"[INFO] Fetching market: {mk}")
        games = _get(api_key, mk)
        if not games:
            continue
        flat = _flatten(games, mk)
        if not flat.empty:
            frames.append(flat)

    if not frames:
        print("[ERROR] No lines returned from The Odds API.", file=sys.stderr)
        return pd.DataFrame(columns=["player", "market", "line", "odds", "book", "game_id"])

    df = pd.concat(frames, ignore_index=True)
    # sanitize types
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return pd.NA

    df["line"] = df["line"].apply(_to_float)
    df["odds"] = df["odds"].apply(_to_float)

    # Final minimal set used by downstream code
    cols = ["player", "market", "line", "odds", "book", "game_id", "commence_time", "home_team", "away_team"]
    return df[cols]


def main():
    parser = argparse.ArgumentParser(description="Fetch sportsbook lines from The Odds API.")
    parser.add_argument("--out", required=True, help="Output CSV path (e.g., inputs/sportsbook_lines.csv)")
    args = parser.parse_args()

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Set ODDS_API_KEY env var (GitHub secret) with your The Odds API key.", file=sys.stderr)
        sys.exit(1)

    df = fetch_all(api_key)
    # fail the job if truly nothing came back
    if df.empty:
        print("ERROR: No lines pulled (empty DataFrame).", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(df):,} rows.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    raise SystemExit(main())
