"""Contract check between data_pipeline_project (project 2) and this project.

Runs the same portfolio model on (a) project 2's quality-passed export and
(b) this project's frozen snapshot, over the window the export covers, and
compares every metric. Writes reports/from_pipeline/contract_check.json.

    python scripts/check_pipeline_export.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality import validate  # noqa: E402
from src.portfolio import simulate, metrics  # noqa: E402


def load(cfg, prices_path, calendar_path, end):
    frame = pd.read_csv(prices_path)
    cal = pd.read_csv(calendar_path).date
    cal = cal[cal <= end]
    frame = frame[frame.date <= end]
    prices, issues = validate(frame, cal, list(cfg["securities"]), cfg["anchor"], end)
    if prices is None:
        raise SystemExit(f"Input gate failed for {prices_path}: {issues.head().to_dict('records')}")
    return prices


def main():
    cfg = json.loads((ROOT / "config/from_pipeline.json").read_text(encoding="utf-8"))
    full = json.loads((ROOT / "config/full.json").read_text(encoding="utf-8"))
    end = cfg["end"]
    a = load(cfg, ROOT / cfg["prices"], ROOT / cfg["calendar"], end)
    b = load(full, ROOT / full["prices"], ROOT / full["calendar"], end)
    rows, worst = [], 0.0
    for bps in cfg["cost_bps"]:
        ra = {n: simulate(a, w, bps) for n, w in cfg["portfolios"].items()}
        rb = {n: simulate(b, w, bps) for n, w in cfg["portfolios"].items()}
        for n in cfg["portfolios"]:
            ma = metrics(
                ra[n]["daily"], ra["Benchmark"]["daily"], cfg["annual_days"], cfg["risk_free_annual"]
            )
            mb = metrics(
                rb[n]["daily"], rb["Benchmark"]["daily"], cfg["annual_days"], cfg["risk_free_annual"]
            )
            for k, v in ma.items():
                if isinstance(v, float) and v == v:
                    d = abs(v - mb[k])
                    worst = max(worst, d)
                    rows.append(
                        {
                            "portfolio": n,
                            "cost_bps": bps,
                            "metric": k,
                            "pipeline": v,
                            "snapshot": mb[k],
                            "abs_diff": d,
                        }
                    )
    tol = 1e-6
    result = {
        "status": "passed" if worst <= tol and len(a) == len(b) else "failed",
        "window_end": end,
        "sessions": len(a) - 1,
        "metrics_compared": len(rows),
        "max_abs_difference": worst,
        "tolerance": tol,
        "note": "Export prices are stored to 6 decimals in the pipeline's landing files; differences come from that rounding.",
    }
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "contract_check_metrics.csv", index=False, float_format="%.12g")
    (out / "contract_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
