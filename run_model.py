#!/usr/bin/env python3
import os, sys, pandas as pd, numpy as np
from engine import price_straights
lines_path="inputs/straights.csv"
if not os.path.exists(lines_path):
    print("FATAL: inputs/straights.csv missing."); sys.exit(1)
lines=pd.read_csv(lines_path)
for c in ["line","over_odds","under_odds"]:
    if c in lines.columns: lines[c]=pd.to_numeric(lines[c], errors="coerce")
s,l=price_straights(lines)
os.makedirs("outputs",exist_ok=True)
s.to_csv("outputs/props_straights.csv",index=False)
l.to_csv("outputs/props_ladders.csv",index=False)
print(f"wrote straights={len(s)} ladders={len(l)}")
