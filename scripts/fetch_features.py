# scripts/fetch_features.py
# Builds the pricing input (inputs/straights.csv) and a rich feature store
# (inputs/features_players.parquet) by merging sportsbook lines with advanced features.

from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path

import nfl_data_py as nfl

# Advanced metrics
from metrics.advanced import build_advanced

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "inputs"

def read_book_lines(path: Path) -> pd.DataFrame:
    p = path if path.exists() else (INP / "sportsbook_lines.csv")
    if not p.exists():
        raise FileNotFoundError("sportsbook_lines.csv not found. If you skip API fetch, upload a CSV to inputs/")
    df = pd.read_csv(p)
    # expected columns: market, team, player, line, over_odds, under_odds, game_id(optional)
    need = {"market", "team", "player", "line"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Lines CSV missing columns: {missing}")
    return df

def load_player_ids() -> pd.DataFrame:
    # map full_name to gsis_id for joins; fallback to fuzzy join if needed
    ppl = nfl.import_players()
    return ppl[["gsis_id", "display_name", "full_name"]].rename(columns={"gsis_id": "player_id"})

def markets_we_support() -> dict:
    return {
        "player_pass_yds": "pass_yds",
        "player_rush_yds": "rush_yds",
        "player_receiving_yds": "rec_yds",
        "player_receptions": "receptions",
    }

def build_priors(seasons: list[int]) -> pd.DataFrame:
    # rolling priors per player-market from nfl_data_py aggregated stats
    # simple + robust: last 5 games mean & std (downcast to keep memory small)
    stats = []
    for season in seasons:
        s = nfl.import_seasonal_passing_stats([season])
        s["market"] = "pass_yds"; s["value"] = s["passing_yards"]; s["player_id"] = s["player_id"]; stats.append(s[["player_id", "season", "market", "value"]])

        r = nfl.import_seasonal_rushing_stats([season])
        r["market"] = "rush_yds"; r["value"] = r["rushing_yards"]; stats.append(r[["player_id", "season", "market", "value"]])

        c = nfl.import_seasonal_receiving_stats([season])
        c["market"] = "rec_yds"; c["value"] = c["receiving_yards"]; stats.append(c[["player_id", "season", "market", "value"]])

        c2 = nfl.import_seasonal_receiving_stats([season]).rename(columns={"receptions": "value"})
        c2["market"] = "receptions"; stats.append(c2[["player_id", "season", "market", "value"]])

    pri = pd.concat(stats, ignore_index=True)
    pri = (pri.groupby(["player_id", "market"], as_index=False)["value"]
              .mean()
              .rename(columns={"value": "prior_mean"}))
    # crude SD floor; we let engine set stronger market floors
    pri["prior_sd"] = np.maximum(np.sqrt(np.abs(pri["prior_mean"])) * 0.75, 8.0)
    return pri

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=0, help="current season (0=auto)")
    ap.add_argument("--weeks", type=str, default="", help="comma weeks (optional)")
    ap.add_argument("--out_lines", type=str, default=str(INP / "straights.csv"))
    ap.add_argument("--features_out", type=str, default=str(INP / "features_players.parquet"))
    ap.add_argument("--book_csv", type=str, default=str(INP / "sportsbook_lines.csv"))
    args = ap.parse_args()

    seasons = []
    if args.season <= 0:
        seasons = [pd.Timestamp.utcnow().year - 1, pd.Timestamp.utcnow().year]
    else:
        seasons = [args.season - 1, args.season]

    # 1) Read sportsbook lines
    lines = read_book_lines(Path(args.book_csv))
    lines = lines.loc[lines["market"].isin(markets_we_support().keys())].copy()

    # 2) Player IDs
    pid = load_player_ids()
    # best-effort name join (display_name or full_name)
    lines = (lines.merge(pid, how="left", left_on="player", right_on="full_name")
                  .fillna(method="ffill"))
    if lines["player_id"].isna().any():
        # fallback join on display_name too
        lines = lines.merge(pid.rename(columns={"full_name":"full_name_2"}),
                            how="left", left_on="player", right_on="display_name")
        lines["player_id"] = lines["player_id"].fillna(lines["player_id_y"])
        lines.drop(columns=[c for c in lines.columns if c.endswith("_y") or c.endswith("_2")], inplace=True)

    # 3) Priors
    pri = build_priors(seasons)

    # 4) Advanced features (team+player)
    team_adv, player_adv = build_advanced(seasons)

    # 5) Merge features
    m = markets_we_support()
    lines["market_key"] = lines["market"].map(m)

    feats = (lines.merge(pri, how="left", left_on=["player_id", "market_key"], right_on=["player_id", "market"])
                  .merge(player_adv, how="left", on="player_id")
                  .merge(team_adv.add_prefix("off_"), how="left", left_on="team", right_on="off_team"))

    # defensive opponent adjustments (if opponent column exists in book csv)
    if "opponent" in feats.columns:
        feats = feats.merge(team_adv.add_prefix("def_"), how="left",
                            left_on="opponent", right_on="def_team")

    # 6) Persist features store
    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.features_out, index=False)

    # 7) Build straights input minimal schema for engine
    need = ["player_id", "player", "team", "market", "line", "over_odds", "under_odds"]
    for c in need:
        if c not in lines.columns:
            lines[c] = np.nan
    lines[["player_id", "player", "team", "market", "line", "over_odds", "under_odds"]].to_csv(args.out_lines, index=False)
    print(f"[fetch_features] wrote: {args.out_lines}")
    print(f"[fetch_features] wrote features: {args.features_out}")

if __name__ == "__main__":
    main()
