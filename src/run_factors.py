"""Factor-model extension: Fama-French 3/5-factor and momentum regressions.

python scripts/fetch_factors.py                      # once, downloads raw files (manual step)
python -m src.run_factors --config config/full.json  # writes reports/factors
"""

import argparse
import hashlib
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

from .quality import validate
from .portfolio import simulate
from .factors import parse_french, regress_table, decompose, rolling_beta, ols_hac, MODELS

ROOT = Path(__file__).resolve().parents[1]
FF5 = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM = "F-F_Momentum_Factor_daily_CSV.zip"
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"


def load_factors(raw):
    manifest = json.loads((raw.parent / "manifest.json").read_text(encoding="utf-8"))
    for f in manifest["files"]:
        if hashlib.sha256((raw / f["file"]).read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError(f"Checksum mismatch: {f['file']}")
    ff5 = parse_french(raw / FF5)
    mom = parse_french(raw / MOM)
    mom.columns = ["Mom"]
    return ff5.join(mom, how="inner"), manifest


def chart_loadings(t, out):
    m = t[
        t.model.eq("FF5+MOM") & t.series.isin(["Main", "Benchmark", "Active (Main - Benchmark)"])
    ].set_index("series")
    cols = MODELS["FF5+MOM"]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(cols))
    for i, (name, color) in enumerate(
        [("Main", BLUE), ("Benchmark", GRAY), ("Active (Main - Benchmark)", ORANGE)]
    ):
        ax.bar(
            x + (i - 1) * 0.26, [m.loc[name, f"beta_{c}"] for c in cols], width=0.24, color=color, label=name
        )
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_title("FF5 + momentum loadings (daily, 2021-2025)", loc="left", fontsize=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "factor_loadings.png", dpi=200)
    plt.close(fig)


def chart_rolling(rb, out):
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(pd.to_datetime(rb.index), rb.Main, color=BLUE, linewidth=1.6, label="Main")
    ax.plot(pd.to_datetime(rb.index), rb.Benchmark, color=GRAY, linewidth=1.6, label="Benchmark")
    ax.set_title("Trailing 252-day market beta (excess returns on Mkt-RF)", loc="left", fontsize=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "rolling_beta.png", dpi=200)
    plt.close(fig)


def z(text):
    """Avoid printing '-0.00' / '+0.00'."""
    return "0.00" if text in ("-0.00", "+0.00") else text


def pdf(out, notes, t, dec, manifest, ver):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, leading=12.5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15)
    h2 = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=8, spaceAfter=4
    )
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor(MUTED))
    doc = SimpleDocTemplate(
        str(out / "factor_report.pdf"),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    s = [
        Paragraph("Factor Exposure Analysis", h1),
        Paragraph(
            "Extension to the eight-ETF portfolio study &middot; daily returns 2021-2025 &middot; independent project",
            small,
        ),
        Paragraph("Method", h2),
        Paragraph(
            "Daily portfolio excess returns (net of the one-month T-bill rate) are regressed on the Fama-French market, size (SMB), value (HML), "
            "profitability (RMW), investment (CMA) and momentum factors from the Kenneth R. French Data Library. OLS with an intercept, "
            "Newey-West standard errors with 5 lags, alphas annualised with 252 sessions. The active return Main minus Benchmark is regressed "
            "on the same factors without the risk-free rate. US factors are applied to all ETFs, including international ones (see limitations).",
            body,
        ),
        Paragraph("Findings", h2),
    ]
    for n in notes:
        s.append(Paragraph("&bull; " + n, body))
    s.append(Paragraph("Loadings (FF5 + momentum)", h2))
    m = t[t.model.eq("FF5+MOM")]
    hdr = ["Series", "Alpha/yr", "t", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "R2"]
    rows = [hdr] + [
        [
            r.series.replace("Active (Main - Benchmark)", "Active"),
            f"{r.alpha_annual:+.2%}",
            f"{r.alpha_t:+.1f}",
        ]
        + [z(f"{r[f'beta_{c}']:+.2f}") for c in MODELS["FF5+MOM"]]
        + [f"{r.r2:.2f}"]
        for _, r in m.iterrows()
    ]
    tab = Table(rows, colWidths=[72, 50, 32, 44, 40, 40, 40, 40, 40, 34], hAlign="LEFT")
    tab.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 7.5),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efea")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#c9c8c2")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    s.append(tab)
    s.append(Spacer(1, 6))
    s.append(Image(str(out / "factor_loadings.png"), width=6.0 * inch, height=2.73 * inch))
    s.append(Image(str(out / "rolling_beta.png"), width=6.0 * inch, height=2.36 * inch))
    s.append(Paragraph("Active return decomposition (annualised)", h2))
    rows = [["Component", "Contribution"]] + [
        [r.component, f"{r.annualised_contribution:+.2%}"] for _, r in dec.iterrows()
    ]
    t2 = Table(rows, colWidths=[160, 80], hAlign="LEFT")
    t2.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efea")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]
        )
    )
    s.append(t2)
    s.append(Paragraph("Verification and limitations", h2))
    s.append(
        Paragraph(
            f"Coefficients match numpy least squares (max difference {ver['checks']['ols_vs_lstsq_max_abs']:.1e}); the decomposition sums to the "
            f"mean active return (difference {ver['checks']['decomposition_sum_abs_error']:.1e}). Factor files: {', '.join(f['file'] for f in manifest['files'])}, "
            f"downloaded {manifest['files'][0]['retrieved_utc'][:10]}, SHA-256 recorded. Limitations: US factors for international ETFs understate "
            "their explanatory power (lower R2 is expected for VEA, VWO, BNDX); bond exposures are not captured by equity factors; daily close "
            "timing differs for foreign markets; five years is a short sample; factor loadings describe, they do not prove cause.",
            body,
        )
    )
    s.append(
        Paragraph(
            "Source: Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
            small,
        )
    )
    doc.build(s)


def run(config):
    cfg = json.loads((ROOT / config).read_text(encoding="utf-8"))
    out = ROOT / "reports" / "factors"
    out.mkdir(parents=True, exist_ok=True)
    factors, manifest = load_factors(ROOT / "data" / "factors" / "raw")
    frame = pd.read_csv(ROOT / cfg["prices"])
    cal = pd.read_csv(ROOT / cfg["calendar"]).date
    prices, _ = validate(frame, cal, list(cfg["securities"]), cfg["anchor"], cfg["end"])
    series = {n: simulate(prices, w, 0)["daily"].net_return for n, w in cfg["portfolios"].items()}
    series = {n: s.set_axis(s.index.strftime("%Y-%m-%d")) for n, s in series.items()}
    etf = prices.pct_change(fill_method=None).iloc[1:]
    etf.index = etf.index.strftime("%Y-%m-%d")
    common = factors.index.intersection(series["Main"].index)
    missing = sorted(set(series["Main"].index) - set(factors.index))
    table = regress_table({**series, **{t: etf[t] for t in etf.columns}}, factors)
    active = (series["Main"] - series["Benchmark"]).rename("active")
    act = regress_table({"Active (Main - Benchmark)": active}, factors, excess=False)
    table = pd.concat([table, act], ignore_index=True)
    table.to_csv(out / "factor_regressions.csv", index=False, float_format="%.10g")
    dec, dec_err = decompose(active, factors, MODELS["FF5+MOM"], excess=False)
    dec.to_csv(out / "active_return_decomposition.csv", index=False, float_format="%.10g")
    rb = pd.DataFrame({n: rolling_beta(series[n], factors) for n in ["Main", "Benchmark"]})
    rb.rename_axis("date").to_csv(out / "rolling_beta.csv", float_format="%.10g")
    # verification: coefficients vs numpy lstsq
    df = pd.concat([series["Main"].rename("r"), factors], axis=1, join="inner").dropna()
    cols = MODELS["FF5+MOM"]
    X = np.column_stack([np.ones(len(df)), df[cols]])
    ref = np.linalg.lstsq(X, (df.r - df.RF).to_numpy(), rcond=None)[0]
    mine = ols_hac(df.r - df.RF, df[cols])["coef"]
    ver = {
        "status": "passed",
        "checks": {
            "ols_vs_lstsq_max_abs": float(np.max(np.abs(ref - mine))),
            "decomposition_sum_abs_error": float(dec_err),
        },
        "sessions_used": len(common),
        "portfolio_sessions_without_factor_data": missing,
    }
    if max(ver["checks"].values()) > 1e-10:
        ver["status"] = "failed"
    (out / "verification_factors.json").write_text(json.dumps(ver, indent=2), encoding="utf-8")
    (out / "run_manifest_factors.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "factor_files": manifest["files"],
                "window": [common.min(), common.max()],
                "hac_lags": 5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    m = table[table.model.eq("FF5+MOM")].set_index("series")
    f3 = table[table.model.eq("FF3")].set_index("series")
    d = dec.set_index("component").annualised_contribution
    intl = m.loc[["VEA", "VWO", "BNDX"], "r2"]
    act = m.loc["Active (Main - Benchmark)"]
    notes = [
        f"Main's US market beta is {m.loc['Main', 'beta_Mkt-RF']:.2f} versus {m.loc['Benchmark', 'beta_Mkt-RF']:.2f} for the benchmark, although both hold "
        f"60% equity: the international sleeve moves less than one-for-one with the US market (VEA {m.loc['VEA', 'beta_Mkt-RF']:.2f}, "
        f"VWO {m.loc['VWO', 'beta_Mkt-RF']:.2f}). The active market beta is {act['beta_Mkt-RF']:+.3f} (t {act['t_Mkt-RF']:+.1f}).",
        f"The active return also loads on size (SMB {act['beta_SMB']:+.3f}, t {act['t_SMB']:+.1f}), value (HML {act['beta_HML']:+.3f}, "
        f"t {act['t_HML']:+.1f}) and, negatively, on profitability (RMW {act['beta_RMW']:+.3f}, t {act['t_RMW']:+.1f}). "
        f"These factors explain {act['r2']:.0%} of daily active-return variation; the rest is likely non-US equity and currency exposure, which US factors do not span.",
        f"Average active return was {d.iloc[-1]:+.2%} a year (arithmetic). The lower market beta cost {d['Mkt-RF']:+.2%} in a strong US market, the small-cap tilt "
        f"{d['SMB']:+.2%} and profitability {d['RMW']:+.2%}, while value added {d['HML']:+.2%}. The remaining alpha of {d['alpha']:+.2%} has t = "
        f"{act['alpha_t']:+.1f}: factor tilts account for about {(d.iloc[-1] - d['alpha']) / d.iloc[-1]:.0%} of the shortfall, and the rest cannot be distinguished from noise.",
        f"US equity factors describe the bond ETFs poorly (R2 BND {m.loc['BND', 'r2']:.2f}, IEF {m.loc['IEF', 'r2']:.2f}, BNDX {m.loc['BNDX', 'r2']:.2f}) and the "
        f"international equity ETFs only partly (VEA {m.loc['VEA', 'r2']:.2f}, VWO {m.loc['VWO', 'r2']:.2f}). Portfolio-level alphas therefore absorb unmodelled "
        f"bond and foreign returns and are not interpreted; the active-return regression, where bonds largely cancel, is the informative one.",
        f"Adding RMW, CMA and momentum to the three-factor model raises the active-return R2 only from {f3.loc['Active (Main - Benchmark)', 'r2']:.2f} to "
        f"{act['r2']:.2f}; size and market explain most of what the US factors can explain.",
    ]
    (out / "findings_factors.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    chart_loadings(table, out)
    chart_rolling(rb, out)
    pdf(out, notes, table, dec, manifest, ver)
    print("\n".join(notes))
    print(f"verification {ver['status']}; sessions {len(common)}; outputs {out}")
    if ver["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/full.json")
    run(p.parse_args().config)
