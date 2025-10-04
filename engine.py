# engine.py
# Prices props using priors + advanced adjustments + Kelly + ladders.

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
INP = ROOT / "inputs"
OUT = ROOT / "outputs"

CONFIG = {
    "n_sims": 20000,
    "kelly_cap": 0.05,        # maximum Kelly fraction
    "min_edge": 0.01,         # require at least 1% edge
    "market_sd_floor": {      # conservative distribution floors (units of stat)
        "pass_yds": 18.0,
        "rush_yds": 12.0,
        "rec_yds": 12.0,
        "receptions": 1.25,
    },
    "usage_weight": 0.35,     # strength of usage (share) adjustments
    "pace_weight": 0.20,      # strength of neutral pace / PROE adjustments
    "def_weight": 0.30,       # strength of opponent EPA-based adjustments
}

MARKET_ALIASES = {
    "player_pass_yds": "pass_yds",
    "player_rush_yds": "rush_yds",
    "player_receiving_yds": "rec_yds",
    "player_receptions": "receptions",
}

def _safe(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def _kelly_fraction(p_true: float, price: float, cap: float) -> float:
    # price is American odds; convert to decimal and do Kelly on EV
    # If price is NaN, return 0
    if np.isnan(price):
        return 0.0
    if price > 0:
        dec = 1 + price / 100.0
    else:
        dec = 1 + 100.0 / abs(price)

    b = dec - 1.0
    q = 1 - p_true
    k = (b * p_true - q) / b
    return max(0.0, min(cap, k))

def _implied_p(american: float) -> float:
    if np.isnan(american):
        return np.nan
    return (100.0 / (american + 100.0)) if american > 0 else (abs(american) / (abs(american) + 100.0))

def _adj_mean(row: pd.Series) -> float:
    mean = _safe(row.get("prior_mean"), 0.0)
    mk = row["market_key"]

    # Usage (receiving: target share & aDOT; rushing: rush/inside-5 share)
    usage_mult = 1.0
    if mk in ["rec_yds", "receptions"]:
        ts = _safe(row.get("target_share_l5"), 0.0)
        adot = _safe(row.get("adot_l5"), 0.0)
        usage_mult *= (1.0 + CONFIG["usage_weight"] * (ts - 0.18))  # 18% ~ avg primary receiver
        usage_mult *= (1.0 + 0.10 * (adot - 8.0) / 8.0)            # nudges if aDOT well above/below avg
    elif mk == "rush_yds":
        rs = _safe(row.get("rush_share_l5"), 0.0)
        i5 = _safe(row.get("i5_share_l5"), 0.0)
        usage_mult *= (1.0 + CONFIG["usage_weight"] * (rs - 0.55))  # lead back ~55% share baseline
        usage_mult *= (1.0 + 0.05 * (i5 - 0.35))                     # goal-line share tiny nudge
    elif mk == "pass_yds":
        # a bit of boost if offense PROE is high
        pass

    # Pace & PROE
    pace_mult = 1.0
    proe = _safe(row.get("off_proe"), 0.0)
    neutral_idx = _safe(row.get("off_neutral_plays_per_game_idx"), 1.0)
    pace_mult *= (1.0 + CONFIG["pace_weight"] * proe)        # +/- from expected pass tendency
    pace_mult *= neutral_idx ** 0.25                         # gentle pace scaling

    # Opponent defense
    def_adj = 1.0
    opp_def = None
    if mk in ["pass_yds", "rec_yds", "receptions"]:
        opp_def = -_safe(row.get("def_def_epa_allowed"))  # negative EPA allowed = tough defense
    elif mk == "rush_yds":
        opp_def = -_safe(row.get("def_def_epa_allowed"))
    if opp_def is not None:
        def_adj *= (1.0 + CONFIG["def_weight"] * opp_def)

    mean *= usage_mult * pace_mult * def_adj
    return max(0.0, mean)

def _adj_sd(row: pd.Series) -> float:
    mk = row["market_key"]
    sd = _safe(row.get("prior_sd"), CONFIG["market_sd_floor"].get(mk, 10.0))
    return float(max(CONFIG["market_sd_floor"].get(mk, 10.0), sd))

def _ladder_targets(mk: str, line: float) -> list[float]:
    if mk in ["rec_yds", "rush_yds", "pass_yds"]:
        step = 15.0 if mk != "pass_yds" else 25.0
    else:
        step = 1.0
    return [line + i * step for i in (1, 2, 3)]

def price_row(row: pd.Series) -> dict:
    mk = row["market_key"]
    line = _safe(row.get("line"), 0.0)
    mean = _adj_mean(row)
    sd = _adj_sd(row)

    # Over/Under probabilities from normal
    p_over = 1.0 - norm.cdf((line - mean) / sd)
    p_under = 1.0 - p_over

    # Kelly suggestions
    k_over = _kelly_fraction(p_over, _safe(row.get("over_odds"), np.nan), CONFIG["kelly_cap"])
    k_under = _kelly_fraction(p_under, _safe(row.get("under_odds"), np.nan), CONFIG["kelly_cap"])

    edge_over = p_over - _implied_p(_safe(row.get("over_odds"), np.nan))
    edge_under = p_under - _implied_p(_safe(row.get("under_odds"), np.nan))

    best_side = "over" if edge_over >= edge_under else "under"
    best_edge = max(edge_over, edge_under)

    return {
        "model_mean": mean,
        "model_sd": sd,
        "p_over": p_over,
        "p_under": p_under,
        "kelly_over": k_over if edge_over >= CONFIG["min_edge"] else 0.0,
        "kelly_under": k_under if edge_under >= CONFIG["min_edge"] else 0.0,
        "best_side": best_side if best_edge >= CONFIG["min_edge"] else "pass",
        "best_edge": float(max(0.0, best_edge)),
    }

def simulate_ladders(row: pd.Series) -> list[dict]:
    mk = row["market_key"]
    mean = _safe(row.get("model_mean"), 0.0)
    sd = _safe(row.get("model_sd"), 10.0)
    if sd <= 0:
        return []

    ladders = []
    for tgt in _ladder_targets(mk, _safe(row.get("line"), 0.0)):
        p_over = 1.0 - norm.cdf((tgt - mean) / sd)
        ladders.append({"alt_line": tgt, "p_over": p_over})
    return ladders

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats_path = INP / "features_players.parquet"
    lines_path = INP / "straights.csv"

    if not feats_path.exists():
        print("[engine] features store not found; run fetch_features first.")
        return
    features = pd.read_parquet(feats_path)

    # Map market -> key used in priors
    features["market_key"] = features["market"].map(MARKET_ALIASES)
    features = features.dropna(subset=["market_key", "line"]).copy()

    # Price straights
    priced = features.apply(price_row, axis=1, result_type="expand")
    out = pd.concat([features.reset_index(drop=True), priced], axis=1)

    cols = [
        "player", "team", "market", "line", "over_odds", "under_odds",
        "model_mean", "model_sd", "p_over", "p_under",
        "kelly_over", "kelly_under", "best_side", "best_edge"
    ]
    out[cols].sort_values("best_edge", ascending=False).to_csv(OUT / "props_straights.csv", index=False)

    # Ladders
    ladd = out.apply(simulate_ladders, axis=1)
    ladd_df = []
    for i, xs in enumerate(ladd):
        if not xs:
            continue
        base = out.loc[i, ["player", "team", "market", "line", "model_mean", "model_sd"]].to_dict()
        for row in xs:
            ladd_df.append(base | row)
    ladd_df = pd.DataFrame(ladd_df)
    if not ladd_df.empty:
        ladd_df.sort_values(["market", "p_over"], ascending=[True, False]).to_csv(OUT / "props_ladders.csv", index=False)

    print("[engine] wrote outputs/props_straights.csv and outputs/props_ladders.csv")

if __name__ == "__main__":
    main()
