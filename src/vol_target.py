"""Volatility targeting as a risk-control overlay on the Main portfolio (no leverage).

At the first session of each month the risky exposure is set to
    w = min(cap, target / forecast_vol)
where forecast_vol is the annualised one-month volatility forecast made with data up to the previous
close. The remaining 1 - w sits in cash earning the one-month T-bill rate (Fama-French RF). Exposure
is held constant within the month (an overlay rebalanced daily to w, as a futures overlay would be);
costs are charged on |change in w| at each monthly reset. Both the original and the managed portfolio
start from Main's zero-cost daily returns, so the comparison isolates the overlay.

The 10% target, the 100% cap and the 5 bps cost were fixed before results were seen; other targets
are reported as sensitivity, not chosen after the fact.
"""
import math

import numpy as np
import pandas as pd


def monthly_exposure(forecast21, first_days, target, cap=1.0):
    """Exposure for each month from the 21-day variance forecast available on its first session."""
    vol = np.sqrt(forecast21.loc[first_days] / 21 * 252)
    return (target / vol).clip(upper=cap).rename("exposure")


def managed_returns(r, rf, exposure_by_month, cost_bps=5.0, initial_exposure=1.0):
    """Daily returns of w * r + (1 - w) * rf with monthly w and costs on |delta w| at each reset."""
    months = pd.Index(r.index).str[:7]
    w_month = pd.Series(exposure_by_month.to_numpy(), index=pd.Index(exposure_by_month.index).str[:7])
    w = pd.Series(months.map(w_month), index=r.index).astype(float)
    if w.isna().any():
        raise ValueError("Exposure missing for some months")
    first = ~pd.Series(months, index=r.index).duplicated()
    prev = w.shift(1).fillna(initial_exposure)
    cost = np.where(first, (w - prev).abs() * cost_bps / 1e4, 0.0)
    out = w * r + (1 - w) * rf - cost
    return out, w, pd.Series(cost, index=r.index)


def stats(r, rf, annual=252):
    """Annualised performance and risk statistics of daily returns."""
    nav = (1 + r).cumprod()
    total = float(nav.iloc[-1] - 1)
    years = len(r) / annual
    ex = r - rf
    dd = nav / np.maximum(1, nav.cummax()) - 1
    monthly_vol = r.groupby(pd.Index(r.index).str[:7]).std(ddof=1) * math.sqrt(annual)
    q = r.quantile(0.05)
    return {"cumulative": total, "annual_return": (1 + total) ** (1 / years) - 1,
            "annual_vol": float(r.std(ddof=1) * math.sqrt(annual)),
            "sharpe": float(ex.mean() / ex.std(ddof=1) * math.sqrt(annual)),
            "max_drawdown": float(dd.min()),
            "worst_month": float(((1 + r).groupby(pd.Index(r.index).str[:7]).prod() - 1).min()),
            "es_95_daily": float(-r[r <= q].mean()),
            "monthly_vol_dispersion": float(monthly_vol.std(ddof=1)),
            "highest_monthly_vol": float(monthly_vol.max())}


def memmel_test(a, b):
    """Jobson-Korkie test with Memmel's correction for equal Sharpe ratios of two return series.
    Returns (z, two-sided p). Positive z: a has the higher Sharpe ratio."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    T = len(a)
    s1 = a.mean() / a.std(ddof=1); s2 = b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    v = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T
    z = (s1 - s2) / math.sqrt(v)
    return float(z), math.erfc(abs(z) / math.sqrt(2))
