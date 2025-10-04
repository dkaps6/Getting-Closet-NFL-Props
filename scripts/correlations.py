# scripts/correlations.py
DEFAULT_R = {
  ("QB_pass_yds","WR_rec_yds"): 0.60,
  ("RB_rush_yds","QB_pass_yds"): -0.35,
  ("WR1_rec_yds","WR2_rec_yds"): 0.20
}

def pairwise_r(a, b, overrides=None):
    if overrides and (a,b) in overrides: return overrides[(a,b)]
    if overrides and (b,a) in overrides: return overrides[(b,a)]
    return DEFAULT_R.get((a,b), DEFAULT_R.get((b,a), 0.0))
