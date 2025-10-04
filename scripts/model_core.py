# scripts/model_core.py
import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_SD = {
    "player_rec_yds": 26.0, "player_receptions": 1.8,
    "player_rush_yds": 23.0, "player_rush_attempts": 3.0,
    "player_pass_yds": 48.0, "player_pass_tds": 0.9
}

def volume_mean(row):
    # very simple scaffold; replace with your shares/plays logic
    # expects row to have model-ready columns (plays, pass_share, player_tgt_share, etc.)
    return row.get("mu_base", 0.0)

def base_sigma(market):
    return DEFAULT_SD.get(market, 20.0)

def pressure_qb_adjust(mu_base, z_opp_pressure, z_opp_epa_pass):
    return mu_base * (1 - 0.35*z_opp_pressure) * (1 - 0.25*z_opp_epa_pass)

def apply_funnel(adj, is_run_funnel=False, is_pass_funnel=False):
    if is_run_funnel:  return adj * 0.97  # ↓ pass-ish
    if is_pass_funnel: return adj * 1.03
    return adj

def widen_sigma(sigma, volatility_flag=False, factor=0.15):
    return sigma * (1 + factor) if volatility_flag else sigma

def over_prob_normal(L, mu, sigma):
    if sigma <= 0: sigma = 1e-6
    z = (L - mu) / sigma
    return 1 - norm.cdf(z)
