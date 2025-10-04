# scripts/calibration.py
import pandas as pd
from scipy.stats import norm

def brier(y_true, p_pred):
    p = p_pred.clip(1e-6, 1-1e-6)
    return ((p - y_true)**2).mean()

def crps_normal(y, mu, sigma):
    # closed form CRPS for Normal
    from numpy import sqrt, pi, exp
    z = (y - mu)/sigma
    return sigma*( z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/3.1415926535**0.5 )

def shrink_mu(mu, mu_mkt, alpha=0.1):
    # μ ← 0.9 μ + 0.1 μ_market
    return 0.9*mu + alpha*mu_mkt
