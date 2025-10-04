# scripts/odds_api.py
import os, time, requests
import pandas as pd

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

MARKETS_GAME = ["h2h","spreads","totals"]
MARKETS_PROPS = [
    "player_pass_yds","player_pass_tds",
    "player_rush_yds","player_rush_attempts",
    "player_rec_yds","player_receptions",
    "player_rush_rec_yds","player_anytime_td"
]

def _get(path, **params):
    key = os.getenv("ODDS_API_KEY")
    if not key: raise RuntimeError("ODDS_API_KEY not set")
    params.update({"apiKey": key})
    r = requests.get(f"{BASE}{path}", params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def fetch_game_lines(regions="us"):
    js = _get("/odds", regions=regions, markets=",".join(MARKETS_GAME),
              oddsFormat="american", dateFormat="iso")
    rows = []
    for ev in js:
        eid = ev["id"]; home = ev["home_team"]; away = ev["away_team"]; start = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for m in bk.get("markets", []):
                mkey = m["key"]
                for o in m.get("outcomes", []):
                    rows.append({
                        "event_id": eid, "start": start, "home": home, "away": away,
                        "book": book, "market": mkey, "name": o.get("name"),
                        "price": o.get("price"), "point": o.get("point")
                    })
    return pd.DataFrame(rows)

def fetch_props_all_events(regions="us", sleep=0.25):
    # First enumerate events so we can hit per-event props endpoint
    evs = _get("/odds", regions=regions, markets="h2h", oddsFormat="american", dateFormat="iso")
    all_rows = []
    for ev in evs:
        eid = ev["id"]; home = ev["home_team"]; away = ev["away_team"]; start = ev.get("commence_time")
        # per-event props
        js = _get(f"/events/{eid}/odds", regions=regions, markets=",".join(MARKETS_PROPS), oddsFormat="american")
        for bk in js.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for m in bk.get("markets", []):
                mkey = m["key"]
                for o in m.get("outcomes", []):
                    all_rows.append({
                        "event_id": eid, "start": start, "home": home, "away": away,
                        "book": book, "market": mkey,
                        "player_name_raw": o.get("description"),
                        "outcome": o.get("name"),              # Over / Under / Yes / No
                        "price": o.get("price"),
                        "point": o.get("point")                 # alt ladders appear as multiple points
                    })
        time.sleep(sleep)
    return pd.DataFrame(all_rows)
