"""Version-2 findings, charts and English PDF addendum."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
EFFECT_COLORS = {"allocation": BLUE, "selection": ORANGE, "interaction": AQUA}


def pp(x):
    return "0.00 pp" if abs(x) < 5e-7 else f"{x * 100:+.2f} pp"


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def findings(f):
    a = f["attributions"]["Main"]
    assets = a["assets"].selection_incl_interaction
    seg = a["by_segment"]
    annual = a["annual"]
    lead = annual[annual.active > 0].index.tolist()
    lag = annual[annual.active < 0].index.tolist()
    eq_name = "Conservative"
    c = f["attributions"][eq_name]
    table, te = f["table"], f["te"]
    vti_te = table.loc["VTI", "pct_of_tracking_error"]
    eq_share = f["seg_share"].get("Equity", float("nan"))
    roll = f["rolling"]
    peak = roll["pct_Fixed income"].idxmax()
    t = f["tests"]
    main = t[t.portfolio.eq("Main")].set_index(["method", "confidence"])
    h99, n99, e99 = main.loc[("historical", 0.99)], main.loc[("normal", 0.99)], main.loc[("ewma", 0.99)]
    pass95 = all(main.loc[(m, 0.95)].kupiec_p >= 0.05 for m in ["historical", "normal", "ewma"])
    return [
        f"Main trailed the 60/40 benchmark by {abs(a['total_portfolio'] - a['total_benchmark']) * 100:.2f} pp cumulatively (2021-2025). "
        f"Brinson-Fachler attribution assigns {pp(a['by_effect']['allocation'])} to allocation and "
        f"{pp(a['by_effect']['selection'] + a['by_effect']['interaction'])} to selection: both portfolios start every month at 60% equity / 40% fixed income, "
        f"so the gap comes from what is held inside each sleeve, not from the stock-bond mix.",
        f"The equity sleeve explains {pp(seg.loc['Equity', 'total'])} of the gap. Relative to holding VTI, the US small-cap ({pp(assets['VB'])}), "
        f"developed international ({pp(assets['VEA'])}) and emerging-market ({pp(assets['VWO'])}) positions all detracted; "
        f"the fixed-income sleeve contributed {pp(seg.loc['Fixed income', 'total'])}.",
        f"The diversified mix helped in {' and '.join(map(str, lead))} and hurt in {', '.join(map(str, lag))}; the result is period-dependent "
        f"and does not by itself show that the diversification choice was wrong.",
        f"For the {eq_name} portfolio (40/60), allocation explains {pp(c['by_effect']['allocation'])} of its {pp(c['total_portfolio'] - c['total_benchmark'])} gap, "
        f"showing the attribution separates a lower equity weight from security choice.",
        f"Equity holdings generate {pct(eq_share, 0)} of Main's ex-post volatility despite a 60% weight. "
        f"The 30% VTI underweight accounts for {pct(vti_te, 0)} of the {pct(te)} tracking error.",
        f"Fixed income's share of rolling 1-year risk peaked at {pct(roll.loc[peak, 'pct_Fixed income'], 0)} in {peak:%b %Y}, "
        f"when the VTI-BND correlation was {roll.loc[peak, 'corr_VTI_BND']:+.2f}; bonds diversified less when stock-bond correlation turned positive.",
        f"1-day 95% VaR is statistically acceptable for all three methods (Kupiec p >= 0.05: {'yes' if pass95 else 'no'}). "
        f"At 99%, all methods saw more exceptions than the {h99.expected_exceptions:.0f} expected over {int(h99.days)} days "
        f"(historical {int(h99.exceptions)}, normal {int(n99.exceptions)}, EWMA {int(e99.exceptions)}); "
        f"EWMA was closest (Kupiec p = {e99.kupiec_p:.3f}), consistent with fat tails and volatility clustering.",
    ]


def _style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c9c8c2")
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.6)
    ax.set_axisbelow(True)


def charts(out, f):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})
    # 1. Main: contribution to active return by holding
    a = f["attributions"]["Main"]["assets"].selection_incl_interaction.sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    cols = [ORANGE if v < 0 else BLUE for v in a]
    ax.barh(a.index, a * 100, color=cols, height=0.6)
    for i, v in enumerate(a):
        label = "0.00" if abs(v) < 5e-7 else f"{v * 100:+.2f}"
        ax.text(
            v * 100 + (0.12 if v >= 0 else -0.12),
            i,
            label,
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color=INK,
        )
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("Contribution to cumulative active return, percentage points", color=MUTED, fontsize=8)
    ax.set_title("Main vs benchmark: selection effect by holding (2021-2025)", loc="left", fontsize=10)
    _style(ax)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    lo = min(a.min() * 100, 0) - 0.9
    ax.set_xlim(lo, max(a.max() * 100, 0) + 0.9)
    fig.tight_layout()
    fig.savefig(out / "attribution_by_holding.png", dpi=200)
    plt.close(fig)

    # 2. Annual effects, small multiples
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, name in zip(axes, ["Main", "Equal", "Conservative"]):
        ann = f["attributions"][name]["annual"]
        x = range(len(ann))
        for k, eff in enumerate(["allocation", "selection", "interaction"]):
            ax.bar(
                [i + (k - 1) * 0.26 for i in x],
                ann[eff] * 100,
                width=0.24,
                color=EFFECT_COLORS[eff],
                label=eff.capitalize(),
            )
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(y)[2:] for y in ann.index])
        ax.set_title(name, loc="left", fontsize=9)
        _style(ax)
    axes[0].set_ylabel("pp of active return", color=MUTED, fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Annual Brinson-Fachler effects vs 60/40 benchmark", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "attribution_annual.png", dpi=200)
    plt.close(fig)

    # 3. Rolling risk share and stock-bond correlation (two panels, one axis each)
    r = f["rolling"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 3.6), sharex=True)
    a1.plot(r.index, r["pct_Fixed income"] * 100, color=BLUE, linewidth=2)
    a1.set_ylabel("% of volatility", color=MUTED, fontsize=8)
    a1.set_title("Main: fixed income share of trailing 1-year volatility", loc="left", fontsize=9)
    a2.plot(r.index, r["corr_VTI_BND"], color=ORANGE, linewidth=2)
    a2.axhline(0, color=GRAY, linewidth=0.8)
    a2.set_ylabel("correlation", color=MUTED, fontsize=8)
    a2.set_title("Trailing 1-year correlation, VTI vs BND daily returns", loc="left", fontsize=9)
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / "rolling_risk.png", dpi=200)
    plt.close(fig)

    # 4. VaR backtest
    fc = f["forecasts"]["Main"]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.plot(fc.index, fc.realized * 100, color="#b5b4ae", linewidth=0.6, label="Realized daily return")
    ax.plot(fc.index, -fc.hist_var_99 * 100, color=BLUE, linewidth=1.4, label="-VaR 99%, historical")
    ax.plot(fc.index, -fc.ewma_var_99 * 100, color=ORANGE, linewidth=1.4, label="-VaR 99%, EWMA")
    hits = fc[fc.realized < -fc.hist_var_99]
    ax.scatter(hits.index, hits.realized * 100, s=12, color=INK, zorder=3, label="Exception (historical)")
    ax.set_ylabel("% per day", color=MUTED, fontsize=8)
    ax.set_title("Main: out-of-sample 1-day 99% VaR backtest", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "var_backtest.png", dpi=200)
    plt.close(fig)


def _table(data, widths, header_bg="#f0efea"):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#c9c8c2")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#c9c8c2")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def pdf(out, cfg, f, notes, ver):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, leading=12.5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, spaceAfter=4)
    h2 = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=8, spaceAfter=4
    )
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor(MUTED))
    doc = SimpleDocTemplate(
        str(out / "attribution_risk_report.pdf"),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    s = []
    s.append(Paragraph("Performance Attribution and Risk Analytics", h1))
    s.append(
        Paragraph(
            "Version 2 addendum to the eight-ETF portfolio study &middot; retrospective simulation, 2021-2025 &middot; independent project",
            small,
        )
    )
    s.append(Spacer(1, 6))
    s.append(Paragraph("Question", h2))
    s.append(
        Paragraph(
            "Version 1 found that the diversified Main portfolio returned 34.81% against 44.68% for a 60% VTI / 40% BND benchmark. "
            "This addendum asks <b>where</b> that gap came from, <b>where the risk</b> sits, and whether standard 1-day VaR models "
            "would have described the portfolio's losses out of sample.",
            body,
        )
    )
    s.append(Paragraph("Method", h2))
    s.append(
        Paragraph(
            "<b>Attribution.</b> Brinson-Fachler by calendar month on two segments (equity, fixed income), linked over 60 months with "
            "Carino smoothing so effects add exactly to the cumulative arithmetic active return. The model rebalances to target at each "
            "month end with no costs, so each month is buy-and-hold and the monthly holdings-based return equals the daily simulation exactly. "
            "Selection plus interaction is split by holding as weight x (holding return - benchmark segment return). "
            "<b>Risk.</b> Euler decomposition of volatility and tracking error at policy weights using the full-period covariance; "
            "a trailing 252-day version at each month end. <b>VaR/ES.</b> 1-day forecasts at 95% and 99% from a 250-day window using "
            "historical simulation (current weights applied to past asset returns), parametric normal, and EWMA (lambda 0.94); "
            f"backtested on {f['manifest']['var_forecast_days']} days from {f['manifest']['first_var_forecast']} with Kupiec "
            "and Christoffersen tests. Every forecast uses only data available the prior evening.",
            body,
        )
    )
    s.append(Paragraph("Key findings", h2))
    for n in notes:
        s.append(Paragraph("&bull; " + n, body))
    s.append(PageBreak())

    s.append(Paragraph("Attribution results", h2))
    rows = [["Portfolio", "Return", "Benchmark", "Active", "Allocation", "Selection", "Interaction"]]
    for name in ["Main", "Equal", "Conservative"]:
        a = f["attributions"][name]
        e = a["by_effect"]
        rows.append(
            [
                name,
                pct(a["total_portfolio"]),
                pct(a["total_benchmark"]),
                pp(a["total_portfolio"] - a["total_benchmark"]),
                pp(e["allocation"]),
                pp(e["selection"]),
                pp(e["interaction"]),
            ]
        )
    s.append(_table(rows, [80, 60, 62, 62, 66, 62, 66]))
    s.append(
        Paragraph(
            "Cumulative 2021-2025, zero-cost scenario. Effects sum exactly to active return (checked to 1e-10).",
            small,
        )
    )
    s.append(Spacer(1, 6))
    s.append(Image(str(out / "attribution_by_holding.png"), width=6.6 * inch, height=3.0 * inch))
    s.append(Image(str(out / "attribution_annual.png"), width=6.9 * inch, height=2.68 * inch))
    s.append(PageBreak())

    s.append(Paragraph("Risk decomposition (Main, policy weights)", h2))
    tbl = f["table"]
    rows = [["Ticker", "Weight", "Active wt", "% of volatility", "% of tracking error"]]
    for t, r in tbl.iterrows():
        rows.append(
            [
                t,
                pct(r.weight, 0),
                f"{r.active_weight * 100:+.0f}%",
                pct(r.pct_of_volatility, 1),
                pct(r.pct_of_tracking_error, 1),
            ]
        )
    rows.append(["Total", "100%", "0%", f"vol {pct(f['sigma'])}", f"TE {pct(f['te'])}"])
    s.append(_table(rows, [60, 60, 60, 90, 100]))
    s.append(Spacer(1, 6))
    s.append(Image(str(out / "rolling_risk.png"), width=6.6 * inch, height=3.6 * inch))
    s.append(PageBreak())

    s.append(Paragraph("VaR backtest (Main)", h2))
    t = f["tests"]
    t = t[t.portfolio.eq("Main")]
    rows = [
        [
            "Method",
            "Conf.",
            "Expected",
            "Exceptions",
            "Kupiec p",
            "Indep. p",
            "Cond. cov. p",
            "Avg VaR",
            "Avg ES",
        ]
    ]
    for _, r in t.iterrows():
        rows.append(
            [
                r.method,
                f"{r.confidence:.0%}",
                f"{r.expected_exceptions:.1f}",
                int(r.exceptions),
                f"{r.kupiec_p:.3f}",
                f"{r.independence_p:.3f}",
                f"{r.conditional_coverage_p:.3f}",
                pct(r.avg_var),
                pct(r.avg_es),
            ]
        )
    s.append(_table(rows, [62, 38, 50, 56, 50, 50, 64, 50, 50]))
    s.append(
        Paragraph(
            "p &lt; 0.05 rejects correct coverage (Kupiec), independence of exceptions (Christoffersen) or both (conditional coverage). "
            "The same tests for the benchmark are in var_backtest.csv.",
            small,
        )
    )
    s.append(Spacer(1, 6))
    s.append(Image(str(out / "var_backtest.png"), width=6.6 * inch, height=3.3 * inch))
    s.append(Paragraph("Verification", h2))
    s.append(
        Paragraph(
            "21 unit tests (11 from version 1, 10 new: hand-computed Brinson example, Carino linking, reconciliation with the daily simulation, "
            "Euler contributions, Kupiec by hand, clustering detection, VaR window quantile and a no-look-ahead perturbation test). "
            f"Independent re-derivations in verification_v2.json: status <b>{ver['status']}</b> at tolerance {ver['absolute_tolerance']:g}.",
            body,
        )
    )
    s.append(Paragraph("Limitations", h2))
    s.append(
        Paragraph(
            "Retrospective, single five-year window chosen in 2026; results are period-dependent. Two-segment attribution only, because the "
            "benchmark holds no international or small-cap sleeve; selection is measured against VTI and BND. Attribution covers the zero-cost "
            "case. Covariance-based risk uses realized data, not a forecasting risk model; no factor model, stress test or liquidity view. "
            "Adjusted prices are a total-return proxy; no taxes, slippage or cash. This is not investment advice.",
            body,
        )
    )
    doc.build(s)
