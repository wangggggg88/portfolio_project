"""Extract a small real-data offline example, preserving provenance."""
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"data/sample";out.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/"config/full.json").read_text())
cfg.update(end="2021-02-26",prices="data/sample/prices.csv",calendar="data/sample/calendar.csv",output="reports/sample",sample=True)
for name in ["prices.csv","calendar.csv"]:
    f=pd.read_csv(ROOT/"data/snapshot"/name)
    f[f.date.le(cfg["end"])].to_csv(out/name,index=False,float_format="%.12g")
meta=json.loads((ROOT/"data/snapshot/manifest.json").read_text())
meta["sample"]="First two months of full snapshot; illustrative execution only, not five-year evidence"
for name in ["prices.csv","calendar.csv"]:meta[name+"_sha256"]=hashlib.sha256((out/name).read_bytes()).hexdigest()
(out/"manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
(ROOT/"config/sample.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
