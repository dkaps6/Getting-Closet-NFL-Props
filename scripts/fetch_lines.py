#!/usr/bin/env python3
import os, csv, argparse, datetime as dt, requests

MARKETS_CORE = ["player_pass_yds","player_rush_yds","player_receiving_yds","player_receptions","player_anytime_td"]

def fetch(api_key, region="us"):
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    r = requests.get(base, params={"apiKey": api_key, "regions": region, "markets": ",".join(MARKETS_CORE), "oddsFormat":"american","dateFormat":"iso"}, timeout=30)
    r.raise_for_status()
    return r.json()

def flatten(raw):
    rows = []
    for g in raw:
        for bk in g.get("bookmakers",[]):
            for mk in bk.get("markets",[]):
                for oc in mk.get("outcomes",[]):
                    rows.append({
                        "player": oc.get("description","").strip(),
                        "team":"", "opp":"",
                        "market": mk.get("key",""),
                        "line": oc.get("line"),
                        "price": oc.get("price"),
                        "side": oc.get("name",""),
                        "book": bk.get("title",""),
                        "game_id": g.get("id",""),
                        "game_time": g.get("commence_time","")
                    })
    return rows

def collapse(rows):
    out = {}
    for r in rows:
        k=(r["player"],r["market"],r["line"],r["book"],r["game_id"])
        rec=out.get(k,{"player":r["player"],"team":"","opp":"","market":r["market"],"line":r["line"],"book":r["book"],"game_id":r["game_id"],"game_time":r["game_time"],"over_odds":None,"under_odds":None})
        s=str(r.get("side","")).lower()
        if "over" in s or s=="yes": rec["over_odds"]=r["price"]
        elif "under" in s or s=="no": rec["under_odds"]=r["price"]
        out[k]=rec
    return list(out.values())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="inputs/sportsbook_lines.csv")
    args=ap.parse_args()
    api=os.environ.get("ODDS_API_KEY","").strip()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hdr=["player","team","opp","market","line","over_odds","under_odds","book","game_id","game_time"]
    if not api:
        with open(args.out,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(hdr); return
    raw=fetch(api); flat=flatten(raw); ou=collapse(flat)
    import csv
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=hdr); w.writeheader(); [w.writerow(r) for r in ou]

if __name__=="__main__": main()
