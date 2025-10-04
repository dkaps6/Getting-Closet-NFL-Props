# scripts/pricing.py
import math

def american_to_prob(odds):
    if odds is None: return None
    o = float(odds)
    return (100.0/(o+100.0)) if o>0 else ( (-o)/( -o+100.0 ) )

def prob_to_american(p):
    p = max(1e-6, min(1-1e-6, p))
    return int(round(100*(p/(1-p)))) if p>=0.5 else int(round(-100*((1-p)/p)))

def devig_two_way(p_over_raw, p_under_raw):
    if p_over_raw is None or p_under_raw is None: return (p_over_raw, p_under_raw)
    denom = p_over_raw + p_under_raw
    if denom <= 0: return (p_over_raw, p_under_raw)
    return (p_over_raw/denom, p_under_raw/denom)

def blend(model_p, market_p_fair, w_model=0.65):
    if market_p_fair is None: return model_p
    return w_model*model_p + (1-w_model)*market_p_fair

def edge_pct(p_blend, p_mkt_fair):
    if p_mkt_fair is None: return None
    return (p_blend - p_mkt_fair) * 100.0

def kelly_fraction(p, price_american, cap=0.05):
    # Kelly on American odds (assume straight bet)
    if p is None or price_american is None: return 0.0
    b = (abs(price_american)/100.0) if price_american<0 else (price_american/100.0)
    q = 1 - p
    f = (b*p - q)/b
    return max(0.0, min(cap, f))

def tier(edge):
    if edge is None: return "RED"
    if edge >= 6: return "ELITE"
    if edge >= 4: return "GREEN"
    if edge >= 1: return "AMBER"
    return "RED"
