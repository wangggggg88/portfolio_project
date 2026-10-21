"""Static charts, self-contained HTML and a three-page research report."""

import base64
import html
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

COLORS = {"Main": "#156f80", "Benchmark": "#e68a37", "Equal": "#6e70a4", "Conservative": "#839642"}
FONT, BOLD = "Helvetica", "Helvetica-Bold"


def charts(out, results, monthly, linked):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
            "svg.hashsalt": "portfolio-project-v1",
        }
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(out / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)

    for column, name, title in [
        ("nav", "wealth", "Growth of $1 | monthly rebalancing | zero cost"),
        ("drawdown", "drawdown", "Drawdown from prior peak | zero cost"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for portfolio, result in results.items():
            d = result["daily"]
            values = d[column]
            # Start chart at initial capital / zero drawdown anchor.
            start = result["anchor"]
            values = pd.concat([pd.Series([1.0 if column == "nav" else 0.0], index=[start]), values])
            ax.plot(values.index, values, label=portfolio, color=COLORS[portfolio], linewidth=1.4)
        ax.set_title(title, loc="left", weight="bold")
        if column == "drawdown":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(ncol=4, frameon=False, fontsize=9)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    weights = results["Main"]["weights_start"]
    ax.stackplot(
        weights.index,
        *[weights[t] for t in weights],
        labels=weights.columns,
        colors=plt.get_cmap("tab20")(np.arange(len(weights.columns))),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1)
    ax.set_title("Main portfolio | beginning-of-session weights", loc="left", weight="bold")
    ax.legend(ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=8)
    save(fig, "weights")
    main = monthly["Main"]
    mat = pd.DataFrame({"year": main.index.year, "month": main.index.month, "value": main.values}).pivot(
        index="year", columns="month", values="value"
    )
    fig, ax = plt.subplots(figsize=(9, 3.4))
    bound = max(0.01, np.nanmax(np.abs(mat.to_numpy())))
    ax.imshow(mat, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(mat.columns)), [str(m) for m in mat.columns])
    ax.set_yticks(range(len(mat.index)), [str(y) for y in mat.index])
    ax.grid(False)
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if np.isfinite(mat.iloc[y, x]):
                ax.text(x, y, f"{mat.iloc[y, x]:.1%}", ha="center", va="center", fontsize=8)
    ax.set_title("Main portfolio | compounded monthly returns", loc="left", weight="bold")
    save(fig, "monthly")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = linked.sort_values()
    ax.barh(x.index, x.values, color=["#156f80" if v >= 0 else "#b45d59" for v in x])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.axvline(0, color="#555", linewidth=0.6)
    ax.set_title("Main portfolio | linked cumulative wealth contributions", loc="left", weight="bold")
    ax.set_xlabel("Percentage points of total return; not benchmark attribution")
    save(fig, "contributions")


def findings(metrics, annual, linked, sensitivity, active_linked):
    m, b = metrics.loc["Main"], metrics.loc["Benchmark"]
    worst_year = annual.Main.idxmin()
    top = linked.idxmax()
    cost10 = sensitivity.loc[(sensitivity.portfolio == "Main") & (sensitivity.cost_bps == 10)].iloc[0]
    weakest = active_linked.idxmin()
    return [
        f"The main portfolio returned {m.total_return:.2%} cumulatively against {b.total_return:.2%} for the benchmark, a gap of {m.cumulative_excess_pp * 100:.2f} pp; the relative wealth return is {m.relative_wealth_return:.2%}.",
        f"Annualized volatility is {m.annual_volatility:.2%} for the main portfolio and {b.annual_volatility:.2%} for the benchmark; maximum drawdowns are {m.max_drawdown:.2%} and {b.max_drawdown:.2%}. Whether risk was lower depends on looking at both measures.",
        f"The main portfolio's worst calendar year was {worst_year}, at {annual.loc[worst_year, 'Main']:.2%} against {annual.loc[worst_year, 'Benchmark']:.2%} for the benchmark. A multi-asset allocation can still lose in the same year.",
        f"{top} made the largest cumulative wealth contribution, {linked[top] * 100:.2f} pp. Contributions are linked through daily NAV and sum to the total return; they are not attribution relative to the benchmark.",
        f"With 10 bps costs the main portfolio returns {cost10.total_return:.2%} cumulatively, {(m.total_return - cost10.total_return) * 100:.2f} pp less than at zero cost. Costs are charged on buys plus sells, including the initial purchase.",
        f"In the wealth-contribution reconciliation of the return gap, {weakest} has the largest negative difference between its main-portfolio and benchmark contributions, {active_linked[weakest] * 100:.2f} pp. This is an arithmetic reconciliation of two portfolios, not Brinson attribution or proof of cause.",
    ]


def html_dashboard(out, cfg, metrics, annual, monthly, notes):
    def img(name):
        raw = base64.b64encode((out / (name + ".png")).read_bytes()).decode()
        return f'<img alt="{name}" src="data:image/png;base64,{raw}">'

    table = metrics[
        [
            "total_return",
            "annual_return",
            "annual_volatility",
            "max_drawdown",
            "sharpe",
            "tracking_error",
            "information_ratio",
        ]
    ].copy()
    for col in table:
        table[col] = table[col].map(
            lambda v: "n.a."
            if pd.isna(v)
            else (f"{v:.2f}" if col in ["sharpe", "information_ratio"] else f"{v:.2%}")
        )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ETF Portfolio Analytics</title>
<style>body{{font-family:system-ui,sans-serif;background:#eef3f5;color:#213441;margin:0}}main{{max-width:1050px;margin:auto;padding:32px}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}h1{{font-size:32px}}h2{{font-size:21px}}p,li{{line-height:1.7}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #ddd}}th:first-child{{text-align:left}}.scroll{{overflow:auto}}small{{color:#546a76}}</style>
<main><h1>ETF Portfolio Performance and Risk Analytics</h1><p>{cfg["start"]} to {cfg["end"]} · USD · monthly rebalancing</p>
<section><h2>Study design</h2><p>A preset 60% equity / 40% bond allocation is compared with a 60% VTI / 40% BND benchmark. The equal-weight portfolio is 50% equity / 50% bonds and the conservative portfolio 40% / 60%, so their risk budgets differ from the benchmark's.</p><p>Adjusted closes serve as a total-return proxy and the risk-free rate is set to zero. The charts and metrics below are at zero cost; cost scenarios are in the report and CSV files. Weights were fixed before any prices were loaded, but the study is retrospective and does not show future performance. Brinson-Fachler attribution, factor regressions and volatility forecasts are in reports/v2, reports/factors and reports/volatility.</p></section>
<section><h2>Findings</h2><ol>{"".join("<li>" + html.escape(n) + "</li>" for n in notes)}</ol></section>
<section><h2>Metrics</h2><div class="scroll">{table.to_html(border=0)}</div><small>Annualized with 252 trading days; standard deviations use ddof=1; zero denominators are shown as n.a.</small></section>
{"".join("<section>" + img(name) + "</section>" for name in ["wealth", "drawdown", "monthly", "weights", "contributions"])}
<section><h2>Monthly returns · all portfolios</h2><div class="scroll">{monthly.map(lambda v: f"{v:.2%}").to_html(border=0)}</div></section>
<section><h2>Limitations</h2><p>Adjusted closes are a vendor total-return proxy that can be revised; dividends and splits were not rebuilt independently. A USD listing does not remove the currency risk of foreign holdings. ETF exposures overlap, and VB is a small-cap tilt relative to VTI. Taxes, slippage and trading limits are not modeled, and costs are scenarios. Results depend on the benchmark and the sample period, and a retrospective study is not out-of-sample evidence.</p><p>Source: <a href="https://finance.yahoo.com/">Yahoo Finance</a>. Raw responses, download time and SHA-256 checksums are recorded in data/snapshot/manifest.json. No data redistribution license is claimed; market data are used for personal research.</p></section></main></html>"""
    (out / "dashboard.html").write_text(body, encoding="utf-8")


def pdf_report(out, cfg, metrics, annual, notes, sensitivity, metadata):
    c = canvas.Canvas(str(out / "portfolio_report.pdf"), pagesize=(595.28, 841.89), invariant=1)
    c.setTitle("Portfolio Performance and Risk Analytics")
    style = ParagraphStyle("body", fontName=FONT, fontSize=9, leading=13, textColor=HexColor("#263e4c"))

    def header(page, subtitle):
        c.setFillColor(HexColor("#156f80"))
        c.rect(0, 810, 595.28, 32, fill=1, stroke=0)
        c.setFillColor(HexColor("#ffffff"))
        c.setFont(BOLD, 11)
        c.drawString(
            38,
            821,
            "ETF Portfolio Performance and Risk Analytics"
            + (" | short offline sample" if cfg.get("sample") else ""),
        )
        c.setFillColor(HexColor("#213441"))
        c.setFont(BOLD, 18)
        c.drawString(38, 776, subtitle)
        c.setFont(FONT, 8)
        c.drawString(38, 26, f"{cfg['start']} / {cfg['end']} | USD | monthly rebalancing | page {page} of 3")

    def fits(y):
        if y < 40:
            raise ValueError("Report content overflows the page")
        return y

    def para(text, y, size=None):
        s = style if size is None else ParagraphStyle("small", parent=style, fontSize=size, leading=size + 4)
        p = Paragraph(text, s)
        _, h = p.wrap(519, 750)
        p.drawOn(c, 38, fits(y - h))
        return y - h - 7

    def title(text, y):
        y -= 7
        c.setFont(BOLD, 11.5)
        c.setFillColor(HexColor("#156f80"))
        c.drawString(38, y, text)
        return y - 16

    def image(name, y, height):
        c.drawImage(
            ImageReader(str(out / (name + ".png"))),
            38,
            fits(y - height),
            width=519,
            height=height,
            preserveAspectRatio=True,
            anchor="c",
        )
        return y - height - 12

    header(1, "Study design and headline results")
    y = para(
        "Question: how did a preset ETF portfolio, diversified across regions and bond types, perform against a US stock-bond benchmark? The weights were not chosen for the best historical result; the focus is verifiable portfolio calculations and a careful reading of the results.",
        749,
    )
    y = para(
        f"Data: {cfg['anchor']} is the initial price anchor and returns cover {cfg['start']} to {cfg['end']}, {int(metrics.loc['Main', 'observations'])} daily returns. Prices are Yahoo Finance adjusted closes, which include the vendor's dividend and split adjustments, used as a total-return proxy. Downloaded {metadata['retrieved_utc'][:10]} (UTC).",
        y,
    )
    y = title("Portfolios", y)
    y = para(
        "Main: VTI 30%, VB 10%, VEA 15%, VWO 5%, BND 20%, IEF 10%, TIP 5%, BNDX 5% (60% equity, 40% bonds).<br/>Benchmark: VTI 60%, BND 40%. Equal: 12.5% each. Conservative: 20%, 5%, 10%, 5%, 30%, 15%, 10%, 5% (same order). All portfolios use the same dates and monthly rebalancing.",
        y,
    )
    y = image("wealth", y, 192)
    y = title("Zero-cost metrics", y)
    headers = ["Portfolio", "Total return", "Annual return", "Annual vol.", "Max drawdown", "Sharpe"]
    xs = [38, 121, 207, 292, 377, 472]
    c.setFont(FONT, 9)
    for x, text in zip(xs, headers):
        c.drawString(x, y, text)
    y -= 20
    for name, row in metrics.iterrows():
        vals = [
            name,
            f"{row.total_return:.2%}",
            f"{row.annual_return:.2%}",
            f"{row.annual_volatility:.2%}",
            f"{row.max_drawdown:.2%}",
            f"{row.sharpe:.2f}",
        ]
        for x, text in zip(xs, vals):
            c.drawString(x, y, text)
        y -= 19
    y = para(
        "Annual return compounds NAV growth over 252 trading days; volatility is the sample standard deviation of daily returns times sqrt(252). Sharpe uses daily excess returns with a zero risk-free rate (a simplifying assumption).",
        y - 4,
    )
    para(notes[0], y)
    c.showPage()
    header(2, "Risk, allocation and return contributions")
    y = image("drawdown", 749, 182)
    y = para(notes[1], y)
    y = para(notes[2], y)
    y = image("contributions", y, 175)
    y = para(notes[3], y)
    y = para(notes[5], y, 8.5)
    y = title("Relative risk and what drives it", y)
    m = metrics.loc["Main"]
    y = para(
        f"Tracking error is {m.tracking_error:.2%} and the information ratio {m.information_ratio:.2f}. Tracking error is the annualized standard deviation of daily return differences against the benchmark; the information ratio is the annualized mean daily active return divided by tracking error.",
        y,
    )
    y = para(
        "The main portfolio holds less US equity than the benchmark and adds international equity, a small-cap tilt and other bond types, so the gap reflects these exposures. This report compares allocations and absolute return contributions only; Brinson-Fachler attribution, which splits the gap into allocation and selection effects, is in reports/v2. The equal-weight and conservative portfolios also have different stock-bond splits, which affects the risk comparison.",
        y,
    )
    para(
        "Linked wealth contribution is the sum of previous-day NAV times each day's asset contribution, applied to the zero-cost model only; it adds up to the cumulative total return. Daily contributions use the previous day's weights, and multi-day contribution percentages are never added directly.",
        y,
    )
    c.showPage()
    header(3, "Method, costs and checks")
    y = para(
        "Portfolio model: r(i,t) = P(i,t)/P(i,t-1) - 1 and r(p,t) = sum of w(i,t-1) r(i,t). Between rebalances, weights drift as w(i,t) = w(i,t-1)(1 + r(i,t))/(1 + r(p,t)). Target weights are set at the month-end close and apply from the next session's return; new weights are never used on the same day.",
        749,
    )
    y = para(
        "Costs: the rate is applied to buys plus sells as a share of pre-trade NAV, the sum of |target weight - drifted weight|, not divided by two; the initial purchase counts as 100%. Costs are deducted before the next day's return: net return = (1 - cost)(1 + gross return) - 1. All holdings shrink in proportion; there is no separate cash or share ledger. There is no rebalance and no liquidation cost on the final day.",
        y,
    )
    y = title("Cost sensitivity: main portfolio", y)
    c.setFont(FONT, 9)
    xs = [38, 146, 262, 388]
    for x, text in zip(xs, ["Cost (bps)", "Total return", "Annual return", "Max drawdown"]):
        c.drawString(x, y, text)
    y -= 20
    for _, row in sensitivity[sensitivity.portfolio.eq("Main")].iterrows():
        for x, text in zip(
            xs,
            [
                str(int(row.cost_bps)),
                f"{row.total_return:.2%}",
                f"{row.annual_return:.2%}",
                f"{row.max_drawdown:.2%}",
            ],
        ):
            c.drawString(x, y, text)
        y -= 19
    y = para(notes[4], y - 5)
    y = image("monthly", y, 148)
    y = title("Checks and limitations", y)
    y = para(
        "Checks are recorded in verification.json: a hand-computed small portfolio, an independent units-based model, SQLite LAG price returns, and consistency of NAV, returns and contributions. Automated tests cover monthly rebalancing, perturbed future prices, costs and zero denominators. The absolute tolerance is 1e-10.",
        y,
        8.5,
    )
    y = para(
        "Limitations: (1) adjusted prices can be revised by the vendor, and dividends and splits were not rebuilt independently; (2) taxes, slippage and trading limits are not modeled, and costs are assumptions; (3) ETF exposures overlap, and a USD listing can still carry foreign currency risk; (4) weights were fixed before the first data pull, but the study is retrospective and does not test out-of-sample performance.",
        y,
        8.5,
    )
    para(
        "Source and reproduction: Yahoo Finance historical prices; the adjustment definition is at in.help.yahoo.com/kb/adjusted-close-sln28256.html. Data are used for personal research and no redistribution license is claimed. Snapshot checksums, configuration and dependency versions are kept with the project; the README explains how to rerun it.",
        y,
        8,
    )
    c.save()
