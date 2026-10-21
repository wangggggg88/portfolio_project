"""Version 2 entry point: performance attribution and risk analytics.

Reads the same frozen snapshot and config as version 1 and writes only to
reports/v2 (version-1 outputs are not touched).

    python -m src.run_v2 --config config/full.json
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from .quality import validate
from .portfolio import simulate, monthly_returns, metrics
from .attribution import attribute
from .risk import decomposition_table, rolling_segment_risk, var_forecasts, backtest
from . import report_v2

ROOT = Path(__file__).resolve().parents[1]
COMPARED = ["Main", "Equal", "Conservative"]


def broad_segments(cfg):
    return {
        t: ("Equity" if "equity" in v["asset_class"].lower() else "Fixed income")
        for t, v in cfg["securities"].items()
    }


def independent_checks(prices, cfg, attributions, results, decomp, forecasts, tests):
    """Re-derive key numbers with separate, plain-Python code paths."""
    tol = cfg["tolerance"]
    checks = {}
    # 1. Monthly buy-and-hold return from month-end prices (stdlib) vs simulate() monthly NAV.
    me = prices.groupby(prices.index.to_period("M")).tail(1)
    p = me.to_numpy().tolist()
    tick = list(prices.columns)
    worst = 0.0
    for name in COMPARED + ["Benchmark"]:
        w = cfg["portfolios"][name]
        hand = [
            sum(w.get(t, 0) * (p[i][j] / p[i - 1][j] - 1) for j, t in enumerate(tick))
            for i in range(1, len(p))
        ]
        sim = monthly_returns(results[name]["daily"]).tolist()
        worst = max(worst, max(abs(a - b) for a, b in zip(hand, sim)))
    checks["monthly_buy_and_hold_vs_simulation_max_abs_error"] = worst
    # 2. Linked effects sum to cumulative active return, and to v1's cumulative_excess_pp.
    worst_sum, worst_v1 = 0.0, 0.0
    for name, a in attributions.items():
        active = a["total_portfolio"] - a["total_benchmark"]
        worst_sum = max(worst_sum, abs(a["by_effect"].sum() - active))
        m = metrics(
            results[name]["daily"], results["Benchmark"]["daily"], cfg["annual_days"], cfg["risk_free_annual"]
        )
        worst_v1 = max(worst_v1, abs(active - m["cumulative_excess_pp"]))
        sel_int = a["by_effect"]["selection"] + a["by_effect"]["interaction"]
        worst_sum = max(worst_sum, abs(a["assets"].selection_incl_interaction.sum() - sel_int))
    checks["linked_effects_vs_active_return_max_abs_error"] = worst_sum
    checks["active_return_vs_v1_cumulative_excess_max_abs_error"] = worst_v1
    # 3. Carino linking recomputed with a plain loop for Main.
    a = attributions["Main"]
    rp, rb = a["monthly_portfolio"].tolist(), a["monthly_benchmark"].tolist()
    tp = math.prod(1 + x for x in rp) - 1
    tb = math.prod(1 + x for x in rb) - 1
    K = (math.log(1 + tp) - math.log(1 + tb)) / (tp - tb)
    eff = a["monthly_effects"].sum(axis=1).tolist()
    linked = sum(
        e * ((math.log(1 + x) - math.log(1 + y)) / (x - y) if x != y else 1 / (1 + x)) / K
        for e, x, y in zip(eff, rp, rb)
    )
    checks["carino_loop_vs_vectorised_abs_error"] = abs(linked - a["by_effect"].sum())
    # 4. Euler risk contributions sum to volatility / tracking error.
    sigma, te, table = decomp
    checks["risk_contributions_sum_vs_volatility_abs_error"] = abs(table.risk_contribution.sum() - sigma)
    checks["active_contributions_sum_vs_tracking_error_abs_error"] = abs(
        table.active_risk_contribution.sum() - te
    )
    # 5. Historical VaR re-derived with stdlib sorting for first and last forecast of Main.
    R = prices.pct_change(fill_method=None).iloc[1:]
    ws = results["Main"]["weights_start"]
    f = forecasts["Main"]
    worst = 0.0
    for t_date in [f.index[0], f.index[-1]]:
        t = R.index.get_loc(t_date)
        w = ws.iloc[t].tolist()
        pnl = sorted(
            sum(a_ * b_ for a_, b_ in zip(row, w)) for row in R.iloc[t - 250 : t].to_numpy().tolist()
        )
        pos = 0.01 * (len(pnl) - 1)
        lo = int(math.floor(pos))
        q = pnl[lo] + (pnl[lo + 1] - pnl[lo]) * (pos - lo)  # linear interpolation, numpy default
        worst = max(worst, abs(-q - f.hist_var_99.loc[t_date]))
    checks["historical_var_stdlib_vs_numpy_abs_error"] = worst
    # 6. Exception count recomputed directly.
    bt = tests[tests.method.eq("historical") & tests.confidence.eq(0.99)].iloc[0]
    hits = sum(1 for r_, v in zip(f.realized, f.hist_var_99) if r_ < -v)
    checks["exception_count_recount_difference"] = abs(hits - int(bt.exceptions))
    status = "passed" if all(v <= tol for v in checks.values()) else "failed"
    if status != "passed":
        raise AssertionError(checks)
    return {"status": status, "absolute_tolerance": tol, "checks": checks}


def run(config):
    path = ROOT / config
    cfg = json.loads(path.read_text(encoding="utf-8"))
    out = ROOT / cfg["output"].replace("reports/full", "reports/v2").replace(
        "reports/sample", "reports/v2_sample"
    )
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / cfg["prices"])
    calendar = pd.read_csv(ROOT / cfg["calendar"]).date
    prices, issues = validate(frame, calendar, list(cfg["securities"]), cfg["anchor"], cfg["end"])
    if prices is None:
        raise ValueError("Input quality failed")
    segments = broad_segments(cfg)
    results = {name: simulate(prices, w, 0) for name, w in cfg["portfolios"].items()}
    bench = cfg["portfolios"]["Benchmark"]

    # ---- attribution ----
    attributions = {}
    summary = []
    for name in COMPARED:
        a = attribute(prices, cfg["portfolios"][name], bench, segments)
        attributions[name] = a
        a["by_segment"].rename_axis("segment").to_csv(
            out / f"attribution_{name}_by_segment.csv", float_format="%.12g"
        )
        a["assets"].rename_axis("ticker").to_csv(out / f"attribution_{name}_assets.csv", float_format="%.12g")
        a["annual"].rename_axis("year").to_csv(out / f"attribution_{name}_annual.csv", float_format="%.12g")
        monthly = a["monthly_effects"].copy()
        monthly.columns = [f"{e}_{s}" for e, s in monthly.columns]
        monthly.insert(0, "benchmark_return", a["monthly_benchmark"])
        monthly.insert(0, "portfolio_return", a["monthly_portfolio"])
        monthly.rename_axis("month").to_csv(out / f"attribution_{name}_monthly.csv", float_format="%.12g")
        summary.append(
            {
                "portfolio": name,
                "portfolio_return": a["total_portfolio"],
                "benchmark_return": a["total_benchmark"],
                "active_return": a["total_portfolio"] - a["total_benchmark"],
                **a["by_effect"].to_dict(),
            }
        )
    pd.DataFrame(summary).to_csv(out / "attribution_summary.csv", index=False, float_format="%.12g")

    # ---- risk ----
    R = prices.pct_change(fill_method=None).iloc[1:]
    decomp = decomposition_table(R, cfg["portfolios"]["Main"], bench, cfg["annual_days"])
    sigma, te, table = decomp
    table.to_csv(out / "risk_decomposition_Main.csv", float_format="%.12g")
    seg_share = table.groupby(pd.Series(segments)).pct_of_volatility.sum()
    rolling = rolling_segment_risk(R, cfg["portfolios"]["Main"], segments, 252, cfg["annual_days"])
    rolling.to_csv(out / "rolling_risk_Main.csv", float_format="%.12g")
    forecasts, tests = {}, []
    for name in ["Main", "Benchmark"]:
        fc = var_forecasts(R, results[name]["weights_start"], results[name]["daily"].gross_return)
        fc.to_csv(out / f"var_forecasts_{name}.csv", float_format="%.12g")
        forecasts[name] = fc
        tests.append(backtest(fc).assign(portfolio=name))
    tests = pd.concat(tests, ignore_index=True)
    tests = tests[["portfolio"] + [c for c in tests.columns if c != "portfolio"]]
    tests.to_csv(out / "var_backtest.csv", index=False, float_format="%.12g")

    # ---- verification & provenance ----
    ver = independent_checks(
        prices, cfg, attributions, results, decomp, forecasts, tests[tests.portfolio.eq("Main")]
    )
    (out / "verification_v2.json").write_text(json.dumps(ver, indent=2), encoding="utf-8")
    import matplotlib

    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prices_sha256": hashlib.sha256((ROOT / cfg["prices"]).read_bytes()).hexdigest(),
        "attribution_period": "calendar month, Brinson-Fachler, Carino linking, zero-cost scenario",
        "var_window_days": 250,
        "ewma_lambda": 0.94,
        "rolling_risk_window_days": 252,
        "first_var_forecast": str(forecasts["Main"].index[0].date()),
        "var_forecast_days": len(forecasts["Main"]),
        "research_type": "Retrospective simulation",
    }
    (out / "run_manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    facts = {
        "sigma": sigma,
        "te": te,
        "table": table,
        "seg_share": seg_share,
        "rolling": rolling,
        "tests": tests,
        "forecasts": forecasts,
        "attributions": attributions,
        "manifest": manifest,
    }
    notes = report_v2.findings(facts)
    (out / "findings_v2.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    report_v2.charts(out, facts)
    report_v2.pdf(out, cfg, facts, notes, ver)
    print(pd.DataFrame(summary).set_index("portfolio").round(4).to_string())
    print(f"Main ex-post vol {sigma:.4f}, TE {te:.4f}; verification {ver['status']}; outputs: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/full.json")
    run(p.parse_args().config)
