"""Explicit refresh only. Offline analysis never calls this module."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def fetch(config, output):
    cfg = json.loads((ROOT / config).read_text(encoding="utf-8"))
    out = ROOT / output
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # Disable inherited broken local proxy; request remains HTTPS with certificate checks.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = int(pd.Timestamp(cfg["anchor"], tz="UTC").timestamp())
    end = int((pd.Timestamp(cfg["end"], tz="UTC") + pd.Timedelta(days=1)).timestamp())
    records, sources = [], []
    for ticker in cfg["securities"]:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}&interval=1d&events=div%2Csplits"
        payload = opener.open(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=40).read()
        dest = raw / f"{ticker}.json"
        if dest.exists():
            raise FileExistsError(f"Snapshot is immutable: {dest}. Choose a new output directory.")
        dest.write_bytes(payload)
        result = json.loads(payload)["chart"]["result"][0]
        if result["meta"]["currency"] != "USD":
            raise ValueError("Non-USD security")
        close = result["indicators"]["quote"][0]["close"]
        adj = result["indicators"]["adjclose"][0]["adjclose"]
        for ts, p, c in zip(result["timestamp"], adj, close):
            date = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
            records.append({"date": date, "ticker": ticker, "adj_close": p, "close": c,
                            "price_type": "adjusted_close", "source": "Yahoo Finance"})
        sources.append({"ticker": ticker, "url": url, "sha256": hashlib.sha256(payload).hexdigest(),
                        "rows": len(adj), "exchange_timezone": result["meta"].get("exchangeTimezoneName")})
        print(f"Downloaded {ticker}: {len(adj)} observations", flush=True)
    prices = pd.DataFrame(records).sort_values(["date", "ticker"])
    prices.to_csv(out / "prices.csv", index=False, float_format="%.12g")
    # Reference sessions are frozen from the broad US-listed ETF VTI, not inferred
    # from intersection across all assets. This detects gaps in the other assets.
    prices.loc[prices.ticker.eq("VTI"), ["date"]].to_csv(out / "calendar.csv", index=False)
    metadata = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "source": "Yahoo Finance chart endpoint",
                "use": "Local personal educational research; no redistribution license asserted",
                "terms": "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html",
                "adjustment": "https://in.help.yahoo.com/kb/adjusted-close-sln28256.html",
                "reference_calendar": "VTI observed sessions; not an independently licensed exchange calendar",
                "configuration_sha256": hashlib.sha256((ROOT / config).read_bytes()).hexdigest(),
                "sources": sources}
    for f in ["prices.csv", "calendar.csv"]:
        metadata[f + "_sha256"] = hashlib.sha256((out / f).read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/full.json")
    p.add_argument("--output", default="data/snapshot")
    a = p.parse_args()
    fetch(a.config, a.output)
