"""One 2x2 summary figure for the README, drawn only from files already in reports/.

    python scripts/build_summary_figure.py      ->  docs/summary.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
OUT = ROOT / "docs" / "summary.png"

BLUE, ORANGE = "#2a78d6", "#eb6834"          # Main / managed  vs  benchmark / original
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def hbars(ax, labels, values, unit):
    y = range(len(values))[::-1]
    ax.barh(list(y), values, color=BLUE, height=0.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(list(y), labels)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    span = max(abs(v) for v in values)
    for yi, v in zip(y, values):
        ax.text(v + (0.03 if v >= 0 else -0.03) * span, yi, f"{v:+.2f}{unit}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    lo, hi = min(0, min(values)), max(0, max(values))
    ax.set_xlim(lo - 0.3 * span, hi + 0.3 * span)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), dpi=150)

    # 1. Growth of $1
    ax = axes[0, 0]
    for name, color in (("Main", BLUE), ("Benchmark", ORANGE)):
        d = pd.read_csv(R / "full" / f"daily_{name}_0bps.csv", parse_dates=["date"])
        label = "Main, 8 ETFs" if name == "Main" else "Benchmark, 60% VTI / 40% BND"
        ax.plot(d["date"], d["nav"], color=color, lw=1.5, label=f"{label}: {d['nav'].iloc[-1] - 1:+.1%}")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_ylabel("Growth of $1", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax, "1  Performance: 9.87 pp behind (cumulative, 2021-25)")

    # 2. Attribution by holding
    a = pd.read_csv(R / "v2" / "attribution_Main_assets.csv")
    a = a[a["selection_incl_interaction"].abs() > 1e-9]
    hbars(axes[0, 1], a["ticker"], (a["selection_incl_interaction"] * 100).tolist(), " pp")
    style(axes[0, 1], "2  Attribution: the gap is selection, not allocation")
    axes[0, 1].set_xlabel("Contribution to cumulative active return, pp (allocation effect = 0)",
                          color=MUTED, fontsize=8)

    # 3. Factor decomposition of annual active return
    f = pd.read_csv(R / "factors" / "active_return_decomposition.csv")
    f = f[~f["component"].str.startswith("total")]
    names = {"alpha": "Alpha (n.s.)", "Mkt-RF": "Market", "SMB": "Size", "HML": "Value",
             "RMW": "Profitability", "CMA": "Investment", "Mom": "Momentum"}
    hbars(axes[1, 0], [names[c] for c in f["component"]], (f["annualised_contribution"] * 100).tolist(), "%")
    style(axes[1, 0], "3  Factors: tilts explain ~55% of -1.48%/yr")
    axes[1, 0].set_xlabel("Annualized contribution to active return, % (Fama-French 5 + momentum)",
                          color=MUTED, fontsize=8)

    # 4. Volatility target drawdown
    ax = axes[1, 1]
    v = pd.read_csv(R / "volatility" / "vol_target_daily_returns.csv", parse_dates=["date"]).set_index("date")
    for col, color, label in (("Original (fully invested)", ORANGE, "Original"),
                              ("Vol target 10%, EWMA 0.94", BLUE, "10% vol target (EWMA)")):
        nav = (1 + v[col]).cumprod()
        dd = nav / nav.cummax().clip(lower=1) - 1
        ax.plot(dd.index, dd * 100, color=color, lw=1.5, label=f"{label}: max {dd.min():.1%}")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Drawdown, %", color=MUTED, fontsize=8)
    style(ax, "4  Volatility target: shallower drawdowns (2022-25)")

    fig.tight_layout(h_pad=2.2, w_pad=2.5)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
