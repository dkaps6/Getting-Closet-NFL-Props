# metrics/advanced.py
# Builds team- and player-level advanced features from nflverse pbp.
# Open-source only (no proprietary feeds).

from __future__ import annotations
import pandas as pd
import numpy as np
import nfl_data_py as nfl

PD_FLOAT = {"float32": np.float32, "float64": np.float64}

def _safe_div(a, b):
    return np.where(b == 0, 0.0, a / np.where(b == 0, 1.0, b))

def load_pbp(seasons: list[int]) -> pd.DataFrame:
    pbp = nfl.import_pbp_data(seasons, downcast=True)  # pinned in requirements
    # Keep regular season plays only
    pbp = pbp.loc[pbp["season_type"].isin(["REG"])].copy()

    # Booleans
    pbp["is_pass"] = (pbp["pass"] == 1).astype(np.int8)
    pbp["is_rush"] = (pbp["rush"] == 1).astype(np.int8)
    pbp["is_play"] = ((pbp["is_pass"] == 1) | (pbp["is_rush"] == 1)).astype(np.int8)

    # Neutral situation filter (simple but effective)
    pbp["neutral"] = (
        (pbp["score_differential"].between(-7, 7, inclusive="both")) &
        (pbp["qtr"].between(1, 3)) &
        (pbp["wp"].between(0.2, 0.8))
    ).astype(np.int8)

    # Useful fields
    for c in ["epa", "air_yards", "passer_player_id", "receiver_player_id",
              "rusher_player_id", "yardline_100", "posteam", "defteam",
              "game_id", "season", "week"]:
        if c not in pbp.columns:
            pbp[c] = np.nan if c in ["epa", "air_yards", "yardline_100"] else ""

    pbp["targets"] = ((pbp["is_pass"] == 1) & pbp["complete_pass"].isin([0, 1])).astype(np.int8)
    pbp["rz_carry"] = ((pbp["is_rush"] == 1) & (pbp["yardline_100"] <= 20)).astype(np.int8)
    pbp["i5_carry"] = ((pbp["is_rush"] == 1) & (pbp["yardline_100"] <= 5)).astype(np.int8)
    pbp["success"] = (pbp["epa"] > 0).astype(np.int8)

    return pbp


def build_team_advanced(seasons: list[int]) -> pd.DataFrame:
    pbp = load_pbp(seasons)

    # League expected pass rate by down & to-go (coarse baseline)
    keys = ["down", "ydstogo"]
    league_tbl = (
        pbp.loc[pbp["is_play"] == 1, keys + ["is_pass"]]
        .dropna(subset=["down", "ydstogo"])
        .assign(ydstogo=lambda d: d["ydstogo"].clip(1, 20))
        .groupby(keys, as_index=False)["is_pass"].mean()
        .rename(columns={"is_pass": "exp_pass_rate"})
    )

    df = (
        pbp.loc[pbp["is_play"] == 1, ["posteam", "down", "ydstogo", "is_pass", "neutral", "epa", "success", "defteam", "game_id"]]
        .dropna(subset=["down", "ydstogo"])
        .assign(ydstogo=lambda d: d["ydstogo"].clip(1, 20))
        .merge(league_tbl, how="left", on=["down", "ydstogo"])
    )

    # Neutral filters for pace proxy (plays/game in neutral)
    neutral = df.loc[df["neutral"] == 1]
    pace_off = (
        neutral.groupby(["posteam", "game_id"], as_index=False)["is_pass"].size()
        .rename(columns={"size": "neutral_plays"})
        .groupby("posteam", as_index=False)["neutral_plays"].mean()
        .rename(columns={"neutral_plays": "neutral_plays_per_game"})
    )

    # PROE (pass rate over expected)
    proe = (
        df.groupby("posteam", as_index=False)[["is_pass", "exp_pass_rate"]].mean()
        .assign(proe=lambda d: d["is_pass"] - d["exp_pass_rate"])
        [["posteam", "proe"]]
    )

    # Offense EPA & Success
    off = (
        df.groupby("posteam", as_index=False)[["epa", "success"]].mean()
        .rename(columns={"epa": "off_epa_play", "success": "off_success"})
    )

    # Defense EPA & Success (allowed)
    ddf = (
        pbp.loc[pbp["is_play"] == 1, ["defteam", "epa", "success"]]
        .groupby("defteam", as_index=False)[["epa", "success"]].mean()
        .rename(columns={"defteam": "team", "epa": "def_epa_allowed", "success": "def_success_allowed"})
    )

    team = (
        proe.merge(pace_off, how="outer", left_on="posteam", right_on="posteam")
            .merge(off, how="outer", on="posteam")
    )
    team = team.rename(columns={"posteam": "team"})
    team = team.merge(ddf, how="left", on="team")

    # Normalize a couple metrics to league index ~ 1.0
    for c in ["neutral_plays_per_game", "off_epa_play", "def_epa_allowed"]:
        mu = team[c].mean()
        if pd.notnull(mu) and mu != 0:
            team[c + "_idx"] = team[c] / mu
        else:
            team[c + "_idx"] = 1.0

    return team


def build_player_advanced(seasons: list[int]) -> pd.DataFrame:
    pbp = load_pbp(seasons)

    # Receiving usage
    rec = pbp.loc[pbp["targets"] == 1, ["season", "week", "game_id", "posteam", "receiver_player_id", "air_yards"]]
    team_targets = rec.groupby(["game_id", "posteam"], as_index=False).size().rename(columns={"size": "team_targets"})
    player_targets = rec.groupby(["game_id", "posteam", "receiver_player_id"], as_index=False).size().rename(columns={"size": "targets"})
    player_ay = rec.groupby(["game_id", "posteam", "receiver_player_id"], as_index=False)["air_yards"].sum()

    p = (
        player_targets.merge(player_ay, how="left", on=["game_id", "posteam", "receiver_player_id"])
        .merge(team_targets, how="left", on=["game_id", "posteam"])
        .assign(target_share=lambda d: _safe_div(d["targets"], d["team_targets"]),
                ay_share=lambda d: _safe_div(d["air_yards"], d["air_yards"].groupby([d["game_id"], d["posteam"]]).transform("sum")),
                adot=lambda d: _safe_div(d["air_yards"], d["targets"]))
    )

    # Rushing usage
    rush = pbp.loc[pbp["is_rush"] == 1, ["game_id", "posteam", "rusher_player_id", "yardline_100"]]
    team_carries = rush.groupby(["game_id", "posteam"], as_index=False).size().rename(columns={"size": "team_carries"})
    player_carries = rush.groupby(["game_id", "posteam", "rusher_player_id"], as_index=False).size().rename(columns={"size": "carries"})
    player_rz = rush.assign(rz=lambda d: (d["yardline_100"] <= 20).astype(np.int8),
                            i5=lambda d: (d["yardline_100"] <= 5).astype(np.int8)) \
                     .groupby(["game_id", "posteam", "rusher_player_id"], as_index=False)[["rz", "i5"]].sum()

    r = (
        player_carries.merge(player_rz, how="left", on=["game_id", "posteam", "rusher_player_id"])
        .merge(team_carries, how="left", on=["game_id", "posteam"])
        .assign(rush_share=lambda d: _safe_div(d["carries"], d["team_carries"]),
                rz_share=lambda d: _safe_div(d["rz"], d["rz"].groupby([d["game_id"], d["posteam"]]).transform("sum")),
                i5_share=lambda d: _safe_div(d["i5"], d["i5"].groupby([d["game_id"], d["posteam"]]).transform("sum")))
    )

    # Collapse to rolling last-5 per player (EMA to weight recent)
    def last5_ema(g, cols, alpha=0.6):
        g = g.sort_values("game_id")
        out = {}
        for c in cols:
            out[c + "_l5"] = g[c].ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        return pd.Series(out)

    rec_feats = (p.groupby("receiver_player_id")
                   .apply(last5_ema, cols=["target_share", "ay_share", "adot"])
                   .reset_index())

    rush_feats = (r.groupby("rusher_player_id")
                    .apply(last5_ema, cols=["rush_share", "rz_share", "i5_share"])
                    .reset_index())

    # Merge receivers/rushers into a single table (players may appear in both)
    rec_feats = rec_feats.rename(columns={"receiver_player_id": "player_id"})
    rush_feats = rush_feats.rename(columns={"rusher_player_id": "player_id"})

    player = pd.merge(rec_feats, rush_feats, how="outer", on="player_id")
    return player


def build_advanced(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (team_df, player_df) with advanced metrics."""
    team = build_team_advanced(seasons)
    player = build_player_advanced(seasons)
    return team, player
