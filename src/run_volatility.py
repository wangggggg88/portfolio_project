"""Volatility-forecasting extension: which model predicts risk best out of sample, and does it fix 99% VaR?

python -m src.run_volatility --config config/full.json      # writes reports/volatility
"""

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from .quality import validate
from .portfolio import simulate
from .risk import kupiec_pof, christoffersen
from .volatility import forecasts, evaluate, var_from_vol, fit_garch, neg_loglik
from .vol_target import monthly_exposure, managed_returns, stats, memmel_test
from .factors import parse_french

ROOT = Path(__file__).resolve().parents[1]
START = 250  # first forecast uses the first 250 returns, same as the version-2 VaR backtest
LABEL = {
    "hist63": "Historical 63d",
    "hist252": "Historical 252d",
    "ewma": "EWMA 0.94",
    "garch": "GARCH(1,1)",
    "gjr": "GJR-GARCH",
    "har": "HAR",
}
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"


def var_table(v):
    """Kupiec / Christoffersen on the common sample where every VaR column is defined."""
    common = v.dropna()
    rows = []
    for c in v.columns[1:]:
        model, method, lvl = c.rsplit("_", 2)
        a = int(lvl) / 100
        hits = (common.realized_return < -common[c]).astype(int)
        n, x = len(hits), int(hits.sum())
        _, p_pof = kupiec_pof(x, n, 1 - a)
        _, p_ind, _ = christoffersen(hits)
        rows.append(
            {
                "model": model,
                "method": method,
                "confidence": a,
                "days": n,
                "expected": n * (1 - a),
                "exceptions": x,
                "kupiec_p": p_pof,
                "independence_p": p_ind,
                "avg_var": float(common[c].mean()),
            }
        )
    return pd.DataFrame(rows), common.index[0], common.index[-1]


def local_optimum_check(r_pct, p, gjr, trials=40, seed=0):
    """No-scipy check: random perturbations of the optimum must not lower the negative log-likelihood."""
    rng = np.random.default_rng(seed)
    s0 = float(np.var(r_pct))
    base = neg_loglik(p["z"], list(r_pct), gjr, s0)
    worse = [
        neg_loglik(np.array(p["z"]) + rng.normal(0, 0.05, len(p["z"])), list(r_pct), gjr, s0)
        for _ in range(trials)
    ]
    return float(min(worse) - base)


TARGET, CAP, COST_BPS = 0.10, 1.0, 5.0  # fixed before results were seen
SENSITIVITY = (0.08, 0.10, 0.12)


def vol_targeting(out, main_r, fr21):
    """Monthly volatility-targeting overlay on Main from 2022 (first full month with forecasts)."""
    rf = parse_french(ROOT / "data" / "factors" / "raw" / "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip").RF
    ev = main_r.loc["2022-01-01":]
    rfe = rf.reindex(ev.index)
    if rfe.isna().any():
        raise ValueError("Risk-free rate missing for some sessions")
    first = pd.Series(ev.index).groupby(pd.Series(ev.index).str[:7]).first().tolist()
    rows, series, exposures = [], {"Original (fully invested)": ev}, {}
    rows.append(
        {
            "strategy": "Original",
            "model": "-",
            "target": np.nan,
            "avg_exposure": 1.0,
            "min_exposure": 1.0,
            "total_cost_bps": 0.0,
            **stats(ev, rfe),
            "sharpe_diff_p": np.nan,
        }
    )
    for model in ("ewma", "gjr"):
        for tgt in SENSITIVITY:
            w = monthly_exposure(fr21[model], first, tgt, CAP)
            r_m, w_d, cost = managed_returns(ev, rfe, w, COST_BPS)
            z, p = memmel_test(r_m - rfe, ev - rfe)
            rows.append(
                {
                    "strategy": "Vol target",
                    "model": model,
                    "target": tgt,
                    "avg_exposure": float(w_d.mean()),
                    "min_exposure": float(w_d.min()),
                    "total_cost_bps": float(cost.sum() * 1e4),
                    **stats(r_m, rfe),
                    "sharpe_diff_p": p,
                }
            )
            if tgt == TARGET:
                series[f"Vol target 10%, {LABEL[model]}"] = r_m
                exposures[model] = w
    table = pd.DataFrame(rows)
    table.to_csv(out / "vol_target_results.csv", index=False, float_format="%.10g")
    pd.DataFrame(exposures).rename_axis("first_session").to_csv(
        out / "vol_target_exposure.csv", float_format="%.10g"
    )
    pd.DataFrame(series).rename_axis("date").to_csv(
        out / "vol_target_daily_returns.csv", float_format="%.10g"
    )
    # chart: exposure (top) and drawdown (bottom), two panels with one axis each
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 3.8), sharex=True)
    for model, color in (("ewma", BLUE), ("gjr", ORANGE)):
        e = exposures[model]
        a1.step(
            pd.to_datetime(e.index),
            e.to_numpy() * 100,
            where="post",
            color=color,
            linewidth=1.5,
            label=LABEL[model],
        )
    a1.set_ylabel("exposure, %", color=MUTED, fontsize=8)
    a1.set_ylim(0, 105)
    a1.set_title("Monthly risky exposure for a 10% volatility target (cap 100%)", loc="left", fontsize=9)
    a1.legend(frameon=False, fontsize=7, loc="lower right")
    for name, color in (
        ("Original (fully invested)", GRAY),
        ("Vol target 10%, EWMA 0.94", BLUE),
        ("Vol target 10%, GJR-GARCH", ORANGE),
    ):
        nav = (1 + series[name]).cumprod()
        a2.plot(
            pd.to_datetime(nav.index),
            (nav / nav.cummax().clip(lower=1) - 1) * 100,
            color=color,
            linewidth=1.3,
            label=name,
        )
    a2.set_ylabel("drawdown, %", color=MUTED, fontsize=8)
    a2.set_title("Drawdown from peak", loc="left", fontsize=9)
    a2.legend(frameon=False, fontsize=7, loc="lower right")
    for ax in (a1, a2):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#ecebe7", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out / "vol_target.png", dpi=200)
    plt.close(fig)
    return table, (ev.index[0], ev.index[-1])


def charts(out, main_fr, losses, vt):
    # 1. Main: 21-day forecast vs realized, annualised vol
    f = main_fr[21].dropna()
    ann = lambda s: np.sqrt(s / 21 * 252) * 100
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    ax.plot(
        pd.to_datetime(f.index),
        ann(f.realized),
        color="#c9c8c2",
        linewidth=1.0,
        label="Realized (next 21 sessions)",
    )
    ax.plot(
        pd.to_datetime(f.index),
        ann(f.hist252),
        color=GRAY,
        linewidth=1.3,
        linestyle="--",
        label="Historical 252d",
    )
    ax.plot(pd.to_datetime(f.index), ann(f.ewma), color=BLUE, linewidth=1.4, label="EWMA")
    ax.plot(pd.to_datetime(f.index), ann(f.gjr), color=ORANGE, linewidth=1.4, label="GJR-GARCH")
    ax.set_ylabel("annualised vol, %", color=MUTED, fontsize=8)
    ax.set_title("Main: one-month volatility forecast vs what followed", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out / "forecast_vs_realized.png", dpi=200)
    plt.close(fig)
    # 2. QLIKE relative to EWMA, median across the 10 series, by horizon
    rel = losses.pivot_table(index=["series", "horizon"], columns="model", values="qlike")
    rel = rel.div(rel["ewma"], axis=0)
    med = rel.groupby(level="horizon").median().T.loc[list(LABEL)]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    y = np.arange(len(med))
    ax.barh(y - 0.18, med[1], height=0.34, color=BLUE, label="1 day")
    ax.barh(y + 0.18, med[21], height=0.34, color=ORANGE, label="21 days")
    for i, (a, b) in enumerate(zip(med[1], med[21])):
        ax.text(a + 0.005, i - 0.18, f"{a:.2f}", va="center", fontsize=7)
        ax.text(b + 0.005, i + 0.18, f"{b:.2f}", va="center", fontsize=7)
    ax.axvline(1, color=GRAY, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in med.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(
        "QLIKE loss relative to EWMA (median of 10 series; below 1 = better)", color=MUTED, fontsize=8
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "qlike_vs_ewma.png", dpi=200)
    plt.close(fig)
    # 3. 99% VaR exceptions: normal vs filtered historical simulation
    t = (
        vt[vt.confidence.eq(0.99)]
        .pivot(index="model", columns="method", values="exceptions")
        .loc[list(LABEL)]
    )
    exp = vt[vt.confidence.eq(0.99)].expected.iloc[0]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    x = np.arange(len(t))
    ax.bar(x - 0.18, t["normal"], width=0.34, color=GRAY, label="Normal quantile")
    ax.bar(x + 0.18, t["fhs"], width=0.34, color=BLUE, label="Filtered historical simulation")
    for i, (a, b) in enumerate(zip(t["normal"], t["fhs"])):
        ax.text(i - 0.18, a + 0.3, str(int(a)), ha="center", fontsize=7)
        ax.text(i + 0.18, b + 0.3, str(int(b)), ha="center", fontsize=7)
    ax.axhline(exp, color=ORANGE, linewidth=1.2, linestyle="--", label=f"Expected ({exp:.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in t.index], fontsize=7.5)
    ax.set_title("Main: 99% one-day VaR exceptions by volatility model", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "var99_exceptions.png", dpi=200)
    plt.close(fig)


def _tbl(rows, widths):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 7.6),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efea")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#c9c8c2")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return t


def pdf(out, notes, main_loss, vt, window, ver, tt=None, tt_window=None, tt_notes=()):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, leading=12.3)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15)
    h2 = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=7, spaceAfter=3
    )
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor(MUTED))
    doc = SimpleDocTemplate(
        str(out / "volatility_report.pdf"),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    s = [
        Paragraph("Volatility Forecasting", h1),
        Paragraph(
            "Extension to the eight-ETF portfolio study &middot; out-of-sample, 2021-12-30 to 2025-12-31 &middot; independent project",
            small,
        ),
        Paragraph("Question", h2),
        Paragraph(
            "Returns are hard to predict, but volatility clusters and is partly predictable. Which model forecasts next-day and next-month "
            "volatility best out of sample, and do better forecasts fix the under-coverage of 99% VaR found in the version-2 backtest?",
            body,
        ),
        Paragraph("Method", h2),
        Paragraph(
            "Ten return series (Main, Benchmark and the eight ETFs). Six models: historical variance over 63 and 252 sessions, EWMA "
            "(lambda 0.94), GARCH(1,1), GJR-GARCH(1,1) with a leverage term, and HAR on daily, weekly and monthly squared returns. "
            "GARCH-type parameters are estimated by Gaussian quasi-maximum likelihood (own Nelder-Mead optimiser) on an expanding window "
            "and re-estimated every 21 sessions; HAR is re-fitted on the same schedule using only targets already observed. Each "
            "forecast for day t uses data up to t-1. Losses: QLIKE (robust to the noisy squared-return proxy) and MSE; Diebold-Mariano "
            "tests against EWMA with Newey-West variance; Mincer-Zarnowitz regressions for bias. VaR: normal quantile versus filtered "
            "historical simulation (past standardized returns scaled by today's forecast).",
            body,
        ),
        Paragraph("Findings", h2),
    ]
    for n in notes:
        s.append(Paragraph("&bull; " + n, body))
    s.append(PageBreak())
    s.append(Paragraph("Main portfolio: forecast accuracy", h2))
    rows = [
        ["Model", "Horizon", "QLIKE", "Rank", "MZ slope", "MZ R2", "Avg forecast vol", "Avg realized vol"]
    ]
    for _, r in main_loss.sort_values(["horizon", "rank_qlike"]).iterrows():
        rows.append(
            [
                LABEL[r.model],
                f"{int(r.horizon)}d",
                f"{r.qlike:.3f}",
                int(r.rank_qlike),
                f"{r.mz_slope:.2f}",
                f"{r.mz_r2:.3f}",
                f"{r.avg_forecast_vol_ann:.2%}",
                f"{r.avg_realized_vol_ann:.2%}",
            ]
        )
    s.append(_tbl(rows, [82, 44, 44, 32, 50, 44, 76, 76]))
    s.append(
        Paragraph(
            "MZ slope = 1 and intercept = 0 indicate unbiased forecasts; slopes below 1 mean forecasts over-react. "
            "Squared daily returns are a very noisy target, so 1-day R2 values are low for every model.",
            small,
        )
    )
    s.append(Spacer(1, 5))
    s.append(Image(str(out / "forecast_vs_realized.png"), width=6.6 * inch, height=2.8 * inch))
    s.append(Image(str(out / "qlike_vs_ewma.png"), width=6.6 * inch, height=2.6 * inch))
    s.append(PageBreak())
    s.append(Paragraph("Main portfolio: 1-day VaR backtest by volatility model", h2))
    rows = [["Model", "Method", "Conf.", "Expected", "Exceptions", "Kupiec p", "Indep. p", "Avg VaR"]]
    for _, r in vt.sort_values(["confidence", "method", "model"]).iterrows():
        rows.append(
            [
                LABEL[r.model],
                "FHS" if r.method == "fhs" else "Normal",
                f"{r.confidence:.0%}",
                f"{r.expected:.1f}",
                int(r.exceptions),
                f"{r.kupiec_p:.3f}",
                f"{r.independence_p:.3f}",
                f"{r.avg_var:.2%}",
            ]
        )
    s.append(_tbl(rows, [82, 44, 34, 48, 54, 46, 46, 48]))
    s.append(
        Paragraph(
            f"Same {int(vt.days.iloc[0])} days ({window[0]} to {window[1]}) for every method. Filtered historical simulation uses up "
            "to 250 past standardized returns; returns before the first forecast are standardized with information available at "
            "that date. p &lt; 0.05 rejects correct coverage (Kupiec) or independence of exceptions.",
            small,
        )
    )
    s.append(Spacer(1, 5))
    s.append(Image(str(out / "var99_exceptions.png"), width=6.6 * inch, height=2.5 * inch))
    if tt is not None:
        s.append(PageBreak())
        s.append(Paragraph("Application: volatility targeting (risk control, not return forecasting)", h2))
        s.append(
            Paragraph(
                f"Each month the risky exposure is set to min(100%, 10% / forecast one-month volatility); the rest earns the T-bill "
                f"rate. Target, cap and a 5 bps cost on exposure changes were fixed in advance; 8% and 12% are sensitivity. "
                f"Period {tt_window[0]} to {tt_window[1]}. Sharpe ratios are in excess of the T-bill rate, which was high in "
                "2023-2025, so they are low for every line.",
                body,
            )
        )
        for n in tt_notes:
            s.append(Paragraph("&bull; " + n, body))
        rows = [
            [
                "Strategy",
                "Target",
                "Avg exp.",
                "Return/yr",
                "Vol",
                "Sharpe",
                "Max DD",
                "Worst mo.",
                "ES 95%",
                "Sharpe p",
            ]
        ]
        for _, r in tt.iterrows():
            name = "Original" if r.strategy == "Original" else LABEL[r.model]
            rows.append(
                [
                    name,
                    "-" if pd.isna(r.target) else f"{r.target:.0%}",
                    f"{r.avg_exposure:.0%}",
                    f"{r.annual_return:.2%}",
                    f"{r.annual_vol:.2%}",
                    f"{r.sharpe:.2f}",
                    f"{r.max_drawdown:.1%}",
                    f"{r.worst_month:.1%}",
                    f"{r.es_95_daily:.2%}",
                    "-" if pd.isna(r.sharpe_diff_p) else f"{r.sharpe_diff_p:.2f}",
                ]
            )
        s.append(_tbl(rows, [70, 36, 42, 50, 40, 38, 44, 46, 40, 42]))
        s.append(
            Paragraph(
                "Sharpe p: Jobson-Korkie test with Memmel's correction against the original. ES 95%: average loss on the worst 5% of days.",
                small,
            )
        )
        s.append(Spacer(1, 4))
        s.append(Image(str(out / "vol_target.png"), width=6.6 * inch, height=3.8 * inch))
    s.append(Paragraph("Verification and limitations", h2))
    s.append(
        Paragraph(
            f"Unit tests recover known GARCH and GJR parameters from simulated data, confirm the optimum is a local minimum, check "
            f"loss functions by hand and confirm no forecast changes when a later return is altered. On the real data, random "
            f"perturbations of every fitted optimum raise the negative log-likelihood (smallest increase {ver['min_loglik_increase']:.2e}). "
            "Limitations: four years of out-of-sample data, dominated by the 2022 rate shock and the April 2025 sell-off; squared "
            "returns are a noisy volatility proxy (intraday data would sharpen the comparison); Gaussian likelihood; the portfolio "
            "series is modelled directly rather than through an asset covariance model; nothing here forecasts the direction of returns.",
            body,
        )
    )
    doc.build(s)


def run(config):
    cfg = json.loads((ROOT / config).read_text(encoding="utf-8"))
    out = ROOT / "reports" / "volatility"
    out.mkdir(parents=True, exist_ok=True)
    prices, _ = validate(
        pd.read_csv(ROOT / cfg["prices"]),
        pd.read_csv(ROOT / cfg["calendar"]).date,
        list(cfg["securities"]),
        cfg["anchor"],
        cfg["end"],
    )
    series = {
        n: simulate(prices, cfg["portfolios"][n], 0)["daily"].gross_return for n in ["Main", "Benchmark"]
    }
    rets = prices.pct_change(fill_method=None).iloc[1:]
    for t in rets.columns:
        series[t] = rets[t]
    losses, dms, params, fr_main = [], [], [], None
    min_increase = float("inf")
    for name, r in series.items():
        r = r.copy()
        r.index = r.index.strftime("%Y-%m-%d")
        fr, hist, warm = forecasts(r, START)
        if name == "Main":
            fr_main, warm_main = fr, warm
            for h, df in fr.items():
                df.rename_axis("date").to_csv(out / f"forecasts_Main_{h}d.csv", float_format="%.10g")
        for h, df in fr.items():
            t, dm = evaluate(df, h)
            losses.append(t.assign(series=name))
            dms.append(dm.assign(series=name))
        params.append(hist.assign(series=name))
        pct = r.to_numpy() * 100
        for gjr in (False, True):
            p = fit_garch(pct, gjr=gjr)
            min_increase = min(min_increase, local_optimum_check(pct, p, gjr))
    losses = pd.concat(losses, ignore_index=True)
    dms = pd.concat(dms, ignore_index=True)
    params = pd.concat(params, ignore_index=True)
    cols = ["series"] + [c for c in losses.columns if c != "series"]
    losses[cols].to_csv(out / "forecast_losses.csv", index=False, float_format="%.10g")
    dms[["series"] + [c for c in dms.columns if c != "series"]].to_csv(
        out / "diebold_mariano.csv", index=False, float_format="%.10g"
    )
    params.to_csv(out / "garch_parameter_history.csv", index=False, float_format="%.10g")
    main_r = series["Main"].copy()
    main_r.index = main_r.index.strftime("%Y-%m-%d")
    v = var_from_vol(fr_main[1], main_r, warm_main)
    v.rename_axis("date").to_csv(out / "var_forecasts_by_model.csv", float_format="%.10g")
    vt, first, last = var_table(v)
    vt.to_csv(out / "var_backtest_by_model.csv", index=False, float_format="%.10g")
    ver = {
        "status": "passed" if min_increase > 0 else "failed",
        "min_loglik_increase": min_increase,
        "series": len(series),
        "first_forecast": fr_main[1].index[0],
        "forecast_days": len(fr_main[1]),
    }
    (out / "verification_volatility.json").write_text(json.dumps(ver, indent=2), encoding="utf-8")
    (out / "run_manifest_volatility.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "start_index": START,
                "refit_every": 21,
                "ewma_lambda": 0.94,
                "horizons": [1, 21],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- findings from the numbers
    ml = losses[losses.series.eq("Main")].set_index(["horizon", "model"])
    best = losses.loc[losses.groupby(["series", "horizon"]).qlike.idxmin()]
    best_counts = best.groupby(["horizon", "model"]).size()
    sig = dms[dms.p_value < 0.05]
    worse_252 = sig[(sig.model == "hist252") & (sig.dm_stat_qlike > 0)]
    better = sig[sig.dm_stat_qlike < 0]
    b1, b21 = ml.loc[1].qlike.idxmin(), ml.loc[21].qlike.idxmin()
    dmm = dms[dms.series.eq("Main")].set_index(["horizon", "model"])
    v99 = vt[vt.confidence.eq(0.99)].set_index(["method", "model"])
    rel = losses.pivot_table(index=["series", "horizon"], columns="model", values="qlike")
    rel = rel.div(rel["ewma"], axis=0)
    notes = [
        (
            f"For Main, {LABEL[b1]} has the lowest QLIKE loss at both horizons, but does not beat EWMA significantly "
            f"(Diebold-Mariano p = {dmm.loc[(1, b1), 'p_value']:.2f} at one day, {dmm.loc[(21, b21), 'p_value']:.2f} at one month)."
            if b1 == b21 and b1 != "ewma"
            else f"For Main, the lowest QLIKE loss is {LABEL[b1]} at one day and {LABEL[b21]} at one month."
        ),
        f"Across all 10 series, a long 252-session historical window is significantly worse than EWMA in {len(worse_252)} of 20 series-horizon "
        f"tests (p &lt; 0.05). Only {len(better)} of {len(dms)} model-vs-EWMA tests show a significant improvement over EWMA: the big gain comes "
        "from reacting to recent volatility at all, not from the choice among conditional models.",
        f"GJR-GARCH, which lets falls raise volatility more than rises, has the lowest loss in {int(best_counts.get((1, 'gjr'), 0))} of 10 series at "
        f"one day and {int(best_counts.get((21, 'gjr'), 0))} at one month; its median loss is {rel['gjr'].groupby(level='horizon').median()[1]:.2f}x "
        f"EWMA (1 day) and {rel['gjr'].groupby(level='horizon').median()[21]:.2f}x (1 month).",
        (
            lambda g, w: f"Asset class matters: GJR-GARCH significantly beats EWMA only for "
            f"{', '.join(sorted(set(g.series)))} (US equity and the VTI-heavy benchmark), consistent with the leverage effect "
            f"being strongest in equities; on bond ETFs the GARCH-type and HAR "
            f"models are significantly worse than EWMA in {len(w)} tests ({', '.join(sorted(set(w.series)))}), so a simple "
            "EWMA is the safer default for fixed income."
        )(
            better[better.model.eq("gjr")],
            sig[
                (sig.dm_stat_qlike > 0)
                & sig.model.isin(["garch", "gjr", "har"])
                & sig.series.isin(["BND", "IEF", "TIP", "BNDX"])
            ],
        ),
        f"Fixing 99% VaR needs both a responsive volatility model and a fat-tailed distribution. Over {int(vt.days.iloc[0])} days "
        f"(expected {v99.expected.iloc[0]:.0f} exceptions): the slow historical windows fail Kupiec's test even with filtered historical "
        f"simulation ({int(v99.loc[('fhs', 'hist63'), 'exceptions'])} and {int(v99.loc[('fhs', 'hist252'), 'exceptions'])} exceptions); "
        f"the conditional models with a normal quantile fail in {int((v99.loc['normal'].loc[['ewma', 'garch', 'gjr', 'har']].kupiec_p < 0.05).sum())} of 4 cases; "
        f"conditional models with filtered historical simulation pass in {int((v99.loc['fhs'].loc[['ewma', 'garch', 'gjr', 'har']].kupiec_p >= 0.05).sum())} of 4 "
        f"(EWMA {int(v99.loc[('fhs', 'ewma'), 'exceptions'])}, GJR {int(v99.loc[('fhs', 'gjr'), 'exceptions'])}).",
        f"One-month Mincer-Zarnowitz slopes are below 1 for every model ({ml.loc[(21, 'hist252'), 'mz_slope']:.2f} for the 252-day window, "
        f"{ml.loc[21].mz_slope.max():.2f} for the best): no model is fully calibrated, and a year-long window says almost nothing about the next "
        "month. Four years containing two sharp episodes is a short sample for ranking models.",
    ]
    tt, tt_window = vol_targeting(out, main_r, fr_main[21])
    o = tt[tt.strategy.eq("Original")].iloc[0]
    e10 = tt[(tt.model == "ewma") & (tt.target == TARGET)].iloc[0]
    g10 = tt[(tt.model == "gjr") & (tt.target == TARGET)].iloc[0]
    tt_notes = [
        f"With EWMA forecasts the overlay held {e10.avg_exposure:.0%} exposure on average and cut volatility from {o.annual_vol:.1%} to "
        f"{e10.annual_vol:.1%}, maximum drawdown from {o.max_drawdown:.1%} to {e10.max_drawdown:.1%} and the worst month from "
        f"{o.worst_month:.1%} to {e10.worst_month:.1%}, while annual return moved from {o.annual_return:.2%} to {e10.annual_return:.2%} "
        f"(costs {e10.total_cost_bps:.0f} bps in total). The Sharpe ratio is essentially unchanged (p = {e10.sharpe_diff_p:.2f}).",
        f"With GJR-GARCH forecasts the same rule earned {g10.annual_return:.2%} a year with a smaller risk reduction (max drawdown "
        f"{g10.max_drawdown:.1%}); its Sharpe ratio is lower but not significantly (p = {g10.sharpe_diff_p:.2f}). A model that forecasts "
        "slightly better on average did not make the better overlay here, a reminder that forecast accuracy and decision value differ.",
        f"A monthly overlay cannot react to a shock inside the month: the highest monthly volatility stays at {e10.highest_monthly_vol:.0%} "
        f"(April 2025). Four years is far too short to claim the overlay adds value; the evidence is about controlling risk, and later "
        "research (Cederburg et al., 2020) finds volatility management rarely improves performance out of sample.",
    ]
    notes.append(
        f"Applied as a 10% volatility target without leverage, EWMA forecasts reduced Main's maximum drawdown from "
        f"{o.max_drawdown:.1%} to {e10.max_drawdown:.1%} with almost the same annual return ({e10.annual_return:.2%} vs "
        f"{o.annual_return:.2%}); details on page 4."
    )
    (out / "findings_volatility.json").write_text(json.dumps(notes + tt_notes, indent=2), encoding="utf-8")
    charts(out, fr_main, losses, vt)
    pdf(out, notes, losses[losses.series.eq("Main")], vt, (first, last), ver, tt, tt_window, tt_notes)
    print("\n".join(notes))
    print(f"verification {ver['status']}; outputs {out}")
    if ver["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/full.json")
    run(p.parse_args().config)
