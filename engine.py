import numpy as np, pandas as pd
from scipy.stats import norm

CONFIG={
  "kelly_cap":0.05,
  "edge_threshold":0.02,
  "sd_min":{"player_pass_yds":25.0,"player_rush_yds":12.0,"player_receiving_yds":12.0,"player_receptions":2.0},
  "ladder":{"player_pass_yds":[-25,0,25,50,75],"player_rush_yds":[-10,0,10,20,30],"player_receiving_yds":[-10,0,10,20,30],"player_receptions":[-2,0,1,2,3]},
  "coeff":{"opp_def_adj":-0.15,"pace_adj":0.10,"injury_adj":-0.20,"wind":-0.06}
}

def prob_to_american(p):
    if p<=0 or p>=1: return np.nan
    return -int(round(100*p/(1-p))) if p>0.5 else int(round(100*(1-p)/p))

def kelly(p, b, cap=0.05):
    q=1-p
    return float(np.clip((p*b - q)/b if b>0 else 0.0, 0.0, cap))

def dec(american):
    if not np.isfinite(american): return np.nan
    a=float(american); return 1 + (a/100.0 if a>0 else 100.0/(-a))

def project_mean(r):
    mu=r.get("prior_mean",np.nan)
    if not np.isfinite(mu): mu=r.get("line",0.0)
    adj=0.0; c=CONFIG["coeff"]; m=str(r.get("market",""))
    adj+=c["opp_def_adj"]*(r.get("opp_def_adj",0.0) or 0.0)
    adj+=c["pace_adj"]*(r.get("pace_adj",0.0) or 0.0)
    adj+=c["injury_adj"]*(r.get("injury_adj",0.0) or 0.0)
    if "pass" in m or "receiv" in m or "reception" in m:
        adj+=c["wind"]*((r.get("weather_wind",0.0) or 0.0)/20.0)
    return float(mu*(1+adj))

def project_sd(r):
    m=str(r.get("market","")); sd=r.get("prior_sd",np.nan)
    return float(max(sd if np.isfinite(sd) else 0.0, CONFIG["sd_min"].get(m,10.0)))

def price_row(row):
    line=row.get("line",np.nan); 
    if not np.isfinite(line): return None
    mu=project_mean(row); sd=project_sd(row)
    p_over=1.0 - norm.cdf((line-mu)/sd); p_under=1-p_over
    do=dec(row.get("over_odds",np.nan)); du=dec(row.get("under_odds",np.nan))
    edge_over= p_over - (1/do if np.isfinite(do) else np.nan)
    edge_under= p_under - (1/du if np.isfinite(du) else np.nan)
    bo="over" if (edge_over if np.isfinite(edge_over) else -1) > (edge_under if np.isfinite(edge_under) else -1) else "under"
    b = (do-1) if bo=="over" else (du-1)
    k = kelly(p_over if bo=="over" else p_under, b, CONFIG["kelly_cap"]) if np.isfinite(b) else 0.0
    return {"proj_mean":mu,"proj_sd":sd,"p_over":p_over,"p_under":p_under,"edge_over":edge_over,"edge_under":edge_under,"best_side":bo,"best_edge":edge_over if bo=='over' else edge_under,"kelly":k,"fair_over":prob_to_american(p_over),"fair_under":prob_to_american(p_under)}

def synth_ladder(row, steps):
    mu=project_mean(row); sd=project_sd(row); m=str(row.get("market",""))
    out=[]
    for d in steps:
        line=(row.get("line",0.0) or 0.0)+float(d)
        p=1.0 - norm.cdf((line-mu)/sd)
        out.append({"player":row.get("player",""),"team":row.get("team",""),"opp":row.get("opp",""),"market":m,"target_line":line,"model_prob_over":p,"model_fair_american":prob_to_american(p),"game_id":row.get("game_id",""),"book":row.get("book","")})
    return out

def price_straights(lines: pd.DataFrame):
    rows=[]; ladders=[]
    for _,r in lines.iterrows():
        pr=price_row(r.to_dict()); 
        if pr is None: continue
        rec={**r.to_dict(),**pr}; rows.append(rec)
        steps=CONFIG["ladder"].get(str(r.get("market","")),[0])
        ladders+=synth_ladder(r, steps)
    s=pd.DataFrame(rows); l=pd.DataFrame(ladders)
    if not s.empty:
        s=s[s["best_edge"].fillna(-1)>=CONFIG["edge_threshold"]].copy()
        s.sort_values(["best_edge","kelly"],ascending=[False,False],inplace=True)
    if not l.empty: l.sort_values(["player","market","target_line"],inplace=True)
    return s,l
