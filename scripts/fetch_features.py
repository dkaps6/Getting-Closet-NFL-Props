#!/usr/bin/env python3
import os, argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try:
    import nfl_data_py as nfl; NFL_OK=True
except Exception as e:
    print("[warn] nfl_data_py import failed:", e); NFL_OK=False

def weekly(season):
    if not NFL_OK: return pd.DataFrame()
    try:
        return nfl.import_weekly_data([season,season-1])
    except Exception as e:
        print("[warn] weekly fetch failed:", e); return pd.DataFrame()

def build_roll(df):
    if df.empty: return pd.DataFrame()
    df.rename(columns={"player_name":"player","recent_team":"team","opponent_team":"opp"}, inplace=True)
    df=df.sort_values(["player","season","week"])
    g=df.groupby("player",group_keys=False)
    def rmean(s): return s.rolling(5, min_periods=2).mean()
    def rstd(s):  return s.rolling(5, min_periods=2).std()
    df["roll_pass_yds_mean"]=g["passing_yards"].apply(rmean); df["roll_pass_yds_std"]=g["passing_yards"].apply(rstd)
    df["roll_rush_yds_mean"]=g["rushing_yards"].apply(rmean); df["roll_rush_yds_std"]=g["rushing_yards"].apply(rstd)
    df["roll_rec_yds_mean"]=g["receiving_yards"].apply(rmean); df["roll_rec_yds_std"]=g["receiving_yards"].apply(rstd)
    df["roll_rec_mean"]=g["receptions"].apply(rmean);       df["roll_rec_std"]=g["receptions"].apply(rstd)
    last=df.groupby("player").tail(1)
    keep=["player","team","opp","roll_pass_yds_mean","roll_rush_yds_mean","roll_rec_yds_mean","roll_rec_mean","roll_pass_yds_std","roll_rush_yds_std","roll_rec_yds_std","roll_rec_std"]
    for c in keep:
        if c not in last.columns: last[c]=np.nan
    for c in last.columns:
        if c.endswith("_std"): last[c]=last[c].fillna(15.0)
        if c.endswith("_mean") or c.endswith("_mean"): last[c]=last[c].fillna(0.0)
    return last[keep]

def merge_features(sb, pri):
    feats=sb.merge(pri, how="left", on="player")
    def pick(row):
        m=str(row["market"])
        if "pass" in m: mu=row.get("roll_pass_yds_mean",np.nan); sd=row.get("roll_pass_yds_std",25.0)
        elif "rush" in m: mu=row.get("roll_rush_yds_mean",np.nan); sd=row.get("roll_rush_yds_std",12.0)
        elif "receiving" in m: mu=row.get("roll_rec_yds_mean",np.nan); sd=row.get("roll_rec_yds_std",12.0)
        elif "receptions" in m: mu=row.get("roll_rec_mean",np.nan); sd=row.get("roll_rec_std",2.0)
        else: mu,sd=np.nan,np.nan
        return pd.Series({"prior_mean":mu,"prior_sd":sd})
    feats[["prior_mean","prior_sd"]]=feats.apply(pick,axis=1)
    feats["opp_def_adj"]=0.0; feats["pace_adj"]=0.0; feats["injury_adj"]=0.0; feats["weather_wind"]=0.0; feats["weather_temp"]=65.0
    return feats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sportsbook",default="inputs/sportsbook_lines.csv")
    ap.add_argument("--out_features",default="inputs/features_players.parquet")
    ap.add_argument("--out_straights",default="inputs/straights.csv")
    a=ap.parse_args()
    season=pd.Timestamp.utcnow().year
    sb=pd.read_csv(a.sportsbook) if os.path.exists(a.sportsbook) else pd.DataFrame(columns=["player","team","opp","market","line","over_odds","under_odds","book","game_id","game_time"])
    pri=build_roll(weekly(season))
    feats=merge_features(sb,pri)
    os.makedirs(os.path.dirname(a.out_features),exist_ok=True)
    feats.to_parquet(a.out_features,index=False)
    sb.to_csv(a.out_straights,index=False)
if __name__=="__main__": main()
