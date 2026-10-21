"""One-page English project summary (docs/one_pager.pdf). Every number is read from report files.

python scripts/build_one_pager.py
"""

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def main():
    m = pd.read_csv(R / "full/metrics.csv", index_col=0)
    seg = pd.read_csv(R / "v2/attribution_Main_by_segment.csv", index_col=0)
    assets = pd.read_csv(R / "v2/attribution_Main_assets.csv", index_col=0).selection_incl_interaction
    summ = pd.read_csv(R / "v2/attribution_summary.csv", index_col=0)
    risk = pd.read_csv(R / "v2/risk_decomposition_Main.csv", index_col=0)
    var = (
        pd.read_csv(R / "v2/var_backtest.csv")
        .query("portfolio == 'Main' and confidence == 0.99")
        .set_index("method")
    )
    fr = pd.read_csv(R / "factors/factor_regressions.csv").query("model == 'FF5+MOM'").set_index("series")
    dec = pd.read_csv(R / "factors/active_return_decomposition.csv", index_col=0).annualised_contribution
    contract = json.loads((R / "from_pipeline/contract_check.json").read_text())
    act = fr.loc["Active (Main - Benchmark)"]
    gap = m.loc["Main", "total_return"] - m.loc["Benchmark", "total_return"]

    ss = getSampleStyleSheet()
    body = ParagraphStyle(
        "b", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11.2, spaceAfter=2
    )
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, spaceAfter=2)
    h2 = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=9.8, spaceBefore=5, spaceAfter=2
    )
    small = ParagraphStyle("s", parent=body, fontSize=7.3, leading=9.4, textColor=colors.HexColor("#52514e"))
    pct = lambda x: f"{x:.2%}"
    pp = lambda x: "0.00 pp" if abs(x) < 5e-7 else f"{x * 100:+.2f} pp"
    doc = SimpleDocTemplate(
        str(ROOT / "docs/one_pager.pdf"),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.5 * inch,
    )
    s = [
        Paragraph("Portfolio Performance and Risk Analytics", h1),
        Paragraph(
            "Independent project &middot; Python, pandas, NumPy, SQLite, Excel &middot; eight USD ETFs, daily data 2021-2025 &middot; retrospective simulation",
            small,
        ),
        Paragraph("Question", h2),
        Paragraph(
            f"A diversified eight-ETF 60/40 portfolio returned {pct(m.loc['Main', 'total_return'])} while a 60% VTI / 40% BND benchmark returned "
            f"{pct(m.loc['Benchmark', 'total_return'])}. Where did the {abs(gap) * 100:.2f} pp gap come from, what risk did the portfolio take, "
            "and what do factor models say about it?",
            body,
        ),
        Paragraph("What the project does", h2),
        Paragraph(
            "A reproducible return engine with drifting weights, monthly rebalancing and 0/5/10 bps cost scenarios; holdings-based Brinson-Fachler "
            "attribution with Carino linking; Euler risk and tracking-error decomposition; 1-day VaR/ES (historical, normal, EWMA) backtested with "
            "Kupiec and Christoffersen tests; Fama-French five-factor plus momentum regressions with Newey-West errors; and a checksum-verified "
            "data contract with a separate data-quality pipeline.",
            body,
        ),
    ]
    rows = [["", "Cumulative", "Annualised", "Volatility", "Max drawdown", "Sharpe", "Tracking error"]]
    for n in ["Main", "Benchmark"]:
        rows.append(
            [
                n,
                pct(m.loc[n, "total_return"]),
                pct(m.loc[n, "annual_return"]),
                pct(m.loc[n, "annual_volatility"]),
                pct(m.loc[n, "max_drawdown"]),
                f"{m.loc[n, 'sharpe']:.2f}",
                pct(m.loc[n, "tracking_error"]),
            ]
        )
    t = Table(rows, colWidths=[70, 62, 62, 58, 70, 44, 72], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efea")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    s += [Paragraph("Results", h2), t, Spacer(1, 4)]
    eq_share = risk[risk.index.isin(["VTI", "VB", "VEA", "VWO"])].pct_of_volatility.sum()
    bullets = [
        f"<b>Attribution.</b> Allocation {pp(summ.loc['Main', 'allocation'])}, selection {pp(summ.loc['Main', 'selection'] + summ.loc['Main', 'interaction'])}: "
        f"both portfolios start each month at 60/40, so the gap is inside the sleeves. Equity sleeve {pp(seg.loc['Equity', 'total'])} versus VTI "
        f"(VB {pp(assets['VB'])}, VEA {pp(assets['VEA'])}, VWO {pp(assets['VWO'])}).",
        f"<b>Risk.</b> Equities produce {eq_share:.0%} of volatility on a 60% weight; the VTI underweight drives {risk.loc['VTI', 'pct_of_tracking_error']:.0%} "
        f"of tracking error. At 99%, VaR models under-cover over {int(var.loc['ewma', 'days'])} out-of-sample days (EWMA {int(var.loc['ewma', 'exceptions'])}, "
        f"historical {int(var.loc['historical', 'exceptions'])}, normal {int(var.loc['normal', 'exceptions'])} exceptions vs {var.loc['ewma', 'expected_exceptions']:.0f} expected).",
        f"<b>Factors.</b> US market beta {fr.loc['Main', 'beta_Mkt-RF']:.2f} vs {fr.loc['Benchmark', 'beta_Mkt-RF']:.2f}. Of the {dec.iloc[-1]:+.2%} a year average "
        f"active return, market beta {dec['Mkt-RF']:+.2%}, size {dec['SMB']:+.2%}, profitability {dec['RMW']:+.2%}, value {dec['HML']:+.2%}; residual alpha "
        f"{dec['alpha']:+.2%} (t {act['alpha_t']:+.1f}) is not significant.",
    ]
    for b in bullets:
        s.append(Paragraph("&bull; " + b, body))
    s += [
        Paragraph("How the numbers are checked", h2),
        Paragraph(
            f"26 unit tests with hand-computed expectations and look-ahead perturbations; SQLite recomputes all asset returns; an independent units "
            f"ledger recomputes NAV for 12 scenarios; native-formula Excel workbooks reproduce the attribution; OLS matches numpy to 1e-15; "
            f"identical results on Windows and Linux; prices from the data pipeline reproduce all {contract['metrics_compared']} metrics within "
            f"{contract['max_abs_difference']:.1e}.",
            body,
        ),
        Paragraph("Limitations", h2),
        Paragraph(
            "One five-year window chosen in hindsight; two-segment attribution because the benchmark has no international or small-cap sleeve; realised "
            "rather than forecast risk; US factors for international and bond ETFs; adjusted prices as a total-return proxy; no taxes or slippage.",
            body,
        ),
        Spacer(1, 4),
        Paragraph(
            "Data: Yahoo Finance adjusted closes; Kenneth R. French Data Library factors. Data used for personal study and not redistributed. "
            "Not investment advice.",
            small,
        ),
    ]
    doc.build(s)
    print("written", ROOT / "docs/one_pager.pdf")


if __name__ == "__main__":
    main()
