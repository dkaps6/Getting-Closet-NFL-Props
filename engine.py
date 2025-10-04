# engine.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


# --------------------------
# Odds / probability helpers
# --------------------------

def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability (vig not removed)."""
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def prob_to_american(p: float) -> float:
    """Convert probability to fair American odds."""
    if not (0 < p < 1):
        return np.nan
    if p >= 0.5:
        return -100.0 * p / (1 - p)
    return 100.0 * (1 - p) / p


def kelly_fraction(p_win: float, american_odds: float, cap: float = 0.05) -> float:
    """
    Kelly fraction for a single binary bet given win prob and American odds.
    Positive = stake fraction, Negative = lay fraction (not used here).
    """
    if pd.isna(p_win) or pd.isna(american_odds):
        return 0.0
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / -american_odds
    q = 1.0 - p_win
    k = (b * p_win - q) / b
    return float(np.clip(k, -cap, cap))


# --------------------------
# Priors and distributions
# --------------------------

@dataclass
class Prior:
    mean: float
    sd: float

def _default_sd_for_market(line: float, market: str) -> float:
    """Fallback spread of player outcomes when no prior sd exists."""
    market = market.lower()
    # These are coarse but robust fallbacks
    if "receptions" in market:
        return max(0.9, 0.35 * max(1.0, abs(line)))
    if "pass" in market:
        return max(10.0, 0.22 * max(1.0, abs(line)))
    if "receiving" in market:
        return max(8.0, 0.28 * max(1.0, abs(line)))
    if "rush" in market:
        return max(7.0, 0.28 * max(1.0, abs(line)))
    return max(6.0, 0.25 * max(1.0, abs(line)))


def _lookup_prior(priors: Optional[pd.DataFrame],
                  player: str,
                  market: str,
                  line: float) -> Prior:
    """Get (mean, sd) prior for a player/market; fall back if missing."""
    if priors is not None and not priors.empty:
        # Expect columns: player, market, mean, sd
        hit = priors[
            (priors["player"].str.lower() == str(player).lower()) &
            (priors["market"].str.lower() == str(market).lower())
        ]
        if not hit.empty:
            r = hit.iloc[0]
            m = float(r.get("mean", np.nan))
            s = float(r.get("sd", np.nan))
            if not np.isnan(m) and not np.isnan(s) and s > 0:
                return Prior(m, s)

    # Fallback: center near the line (so we’re neutral) with a reasonable sd
    return Prior(mean=float(line), sd=_default_sd_for_market(line, market))


def _prob_over_from_prior(line: float, prior: Prior) -> float:
    """P(Outcome > line) under Normal(prior.mean, prior.sd)."""
    if prior.sd <= 0:
        return 0.5
    z = (line - prior.mean) / prior.sd
    return float(1.0 - norm.cdf(z))


# --------------------------
# Market parsing
# --------------------------

def _market_kind(market: str) -> str:
    """Classify market into 'ou' or 'td' (anytime TD)."""
    m = market.lower().strip()
    if "anytime" in m and "td" in m:
        return "td"
    # All others we treat as normal over/under (yards/receptions, etc.)
    return "ou"


# --------------------------
# Pricing functions
# --------------------------

def price_straights(lines: pd.DataFrame,
                    priors: Optional[pd.DataFrame] = None,
                    bankroll_cap: float = 0.05) -> pd.DataFrame:
    """
    Price straight player props (O/U and anytime TD) and compute Kelly stakes.

    Required columns in `lines` (case-insensitive accepted):
      - player
      - market   (e.g., player_rush_yds, player_receiving_yds, player_receptions,
                  player_pass_yds, player_anytime_td)
      - line     (numeric; ignored for TD)
      - over_odds (American)
      - under_odds (American)  (ignored for TD)

    Returns a DataFrame with:
      player, market, line, over_odds, under_odds,
      p_over, p_under, kelly_over, kelly_under, pick, stake
    """
    if lines is None or lines.empty:
        return pd.DataFrame()

    # Normalize column names we rely on
    cols = {c.lower(): c for c in lines.columns}
    def col(name: str) -> str:
        for k, v in cols.items():
            if k == name:
                return v
        return name  # fall back, will raise if missing later

    req = ["player", "market", "line"]
    for r in req:
        if r not in cols:
            raise ValueError(f"Required column '{r}' missing from lines CSV.")

    # optional but preferred
    over_col = cols.get("over_odds")
    under_col = cols.get("under_odds")

    out = []
    for _, row in lines.iterrows():
        player = str(row[col("player")])
        market = str(row[col("market")])
        mkind = _market_kind(market)

        over_odds = float(row[over_col]) if over_col in row and not pd.isna(row[over_col]) else np.nan
        under_odds = float(row[under_col]) if under_col in row and not pd.isna(row[under_col]) else np.nan

        if mkind == "td":
            # For TD we use a prior rate if provided, else neutral-ish default
            # Expect priors row like: market='player_anytime_td', mean=<p>, sd unused
            prior = None
            if priors is not None and not priors.empty:
                hit = priors[
                    (priors["player"].str.lower() == player.lower()) &
                    (priors["market"].str.lower() == market.lower())
                ]
                if not hit.empty and "mean" in hit.columns:
                    p = float(hit.iloc[0]["mean"])
                    if 0 < p < 1:
                        prior = p
            p_over = prior if prior is not None else 0.33  # fallback TD rate
            k_over = kelly_fraction(p_over, over_odds, cap=bankroll_cap)
            out.append(
                dict(player=player, market=market, line=np.nan,
                     over_odds=over_odds, under_odds=np.nan,
                     p_over=p_over, p_under=1 - p_over,
                     kelly_over=k_over, kelly_under=0.0,
                     pick="over" if k_over > 0 else "pass",
                     stake=max(0.0, k_over))
            )
            continue

        # Over/Under markets
        line_val = float(row[col("line")]) if not pd.isna(row[col("line")]) else np.nan
        prior = _lookup_prior(priors, player, market, line_val)
        p_over = _prob_over_from_prior(line_val, prior)
        p_under = 1.0 - p_over

        k_over = kelly_fraction(p_over, over_odds, cap=bankroll_cap)
        k_under = kelly_fraction(p_under, under_odds, cap=bankroll_cap)

        # Choose the better positive stake, else pass
        pick = "pass"
        stake = 0.0
        if k_over > k_under and k_over > 0:
            pick, stake = "over", k_over
        elif k_under > k_over and k_under > 0:
            pick, stake = "under", k_under

        out.append(
            dict(player=player, market=market, line=line_val,
                 over_odds=over_odds, under_odds=under_odds,
                 p_over=p_over, p_under=p_under,
                 kelly_over=k_over, kelly_under=k_under,
                 pick=pick, stake=stake)
        )

    res = pd.DataFrame(out)
    # Sort by stake descending for convenience
    if not res.empty and "stake" in res.columns:
        res = res.sort_values("stake", ascending=False).reset_index(drop=True)
    return res

