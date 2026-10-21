"""Daily return / drifting-weight model; no share or execution-price ledger."""

import numpy as np
import pandas as pd


def simulate(prices, target, cost_bps=0):
    weights = np.array([target.get(t, 0.0) for t in prices.columns], dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-12):
        raise ValueError("Long-only weights must sum to one")
    if cost_bps < 0 or cost_bps >= 10000:
        raise ValueError("Invalid cost")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Invalid returns")
    w = weights.copy()
    nav = 1.0
    # Initial funding charged on 100% purchases. Every subsequent monthly
    # reset charges bps * SUM(abs(target - drifted)), i.e. buys PLUS sells.
    pending_turnover = 1.0
    rows, starts, ends, contributions, trades = [], [], [], [], []
    for i, (date, asset_returns) in enumerate(returns.iterrows()):
        starts.append(w.copy())
        c = w * asset_returns.to_numpy()
        gross = float(c.sum())
        fee = pending_turnover * cost_bps / 10000
        net = (1 - fee) * (1 + gross) - 1
        nav *= 1 + net
        contributions.append(c)
        rows.append([gross, net, nav, fee, pending_turnover])
        w = w * (1 + asset_returns.to_numpy()) / (1 + gross)
        ends.append(w.copy())
        pending_turnover = 0.0
        # Rebalance at the previous month's final observed session close;
        # target exposure is effective for the next session's close-to-close return.
        if i + 1 < len(returns) and date.to_period("M") != returns.index[i + 1].to_period("M"):
            pending_turnover = float(np.abs(weights - w).sum())
            trades.append([date, returns.index[i + 1], pending_turnover])
            w = weights.copy()
    daily = pd.DataFrame(
        rows,
        index=returns.index,
        columns=["gross_return", "net_return", "nav", "cost_fraction", "traded_fraction"],
    )
    # Include initial NAV in running peak so first-session loss is a drawdown.
    daily["drawdown"] = daily.nav / np.maximum(1.0, daily.nav.cummax()) - 1
    return {
        "daily": daily,
        "weights_start": pd.DataFrame(starts, index=returns.index, columns=prices.columns),
        "weights_end": pd.DataFrame(ends, index=returns.index, columns=prices.columns),
        "contributions": pd.DataFrame(contributions, index=returns.index, columns=prices.columns),
        "rebalances": pd.DataFrame(trades, columns=["formation_date", "effective_date", "traded_fraction"]),
    }


def metrics(daily, benchmark, annual_days=252, rf_annual=0.0):
    r = daily.net_return
    active = r - benchmark.net_return
    rf_daily = (1 + rf_annual) ** (1 / annual_days) - 1
    excess = r - rf_daily
    vol = float(r.std(ddof=1) * np.sqrt(annual_days))
    exvol = float(excess.std(ddof=1) * np.sqrt(annual_days))
    te = float(active.std(ddof=1) * np.sqrt(annual_days))
    ratio = lambda a, b: float(a / b) if np.isfinite(b) and abs(b) > 1e-14 else np.nan
    total = float(daily.nav.iloc[-1] - 1)
    bench_total = float(benchmark.nav.iloc[-1] - 1)
    return {
        "observations": len(r),
        "total_return": total,
        "annual_return": float((1 + total) ** (annual_days / len(r)) - 1),
        "annual_volatility": vol,
        "max_drawdown": float(daily.drawdown.min()),
        "sharpe": ratio(excess.mean() * annual_days, exvol),
        "tracking_error": te,
        "information_ratio": ratio(active.mean() * annual_days, te),
        "cumulative_excess_pp": total - bench_total,
        "relative_wealth_return": float((1 + total) / (1 + bench_total) - 1),
        "traded_fraction_sum": float(daily.traded_fraction.sum()),
    }


def monthly_returns(daily):
    return (1 + daily.net_return).groupby(daily.index.to_period("M")).prod() - 1


def linked_contributions(result):
    """Exact gross wealth contribution: sum(NAV_before_day * daily contribution).
    Valid for zero-cost run. This is cumulative wealth contribution, not Brinson.
    """
    nav_before = result["daily"].nav.shift(1, fill_value=1.0)
    return result["contributions"].mul(nav_before, axis=0).sum()
