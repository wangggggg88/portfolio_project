"""Risk decomposition and 1-day VaR / Expected Shortfall with out-of-sample backtests.

Everything here uses only information available at the forecast time:
the forecast for session t uses asset returns up to t-1 and the drifted
weights at the start of t (weights_start from simulate()).
"""

import math
from statistics import NormalDist
import numpy as np
import pandas as pd


# ---------- risk decomposition ----------


def risk_contributions(cov, w):
    """Euler decomposition of portfolio volatility: ctr_i = w_i (Cov w)_i / sigma; sums to sigma."""
    w = np.asarray(w, dtype=float)
    cov = np.asarray(cov, dtype=float)
    sigma = math.sqrt(float(w @ cov @ w))
    if sigma <= 0:
        raise ValueError("Zero portfolio volatility")
    mctr = cov @ w / sigma
    ctr = w * mctr
    return sigma, mctr, ctr


def decomposition_table(asset_returns, w_port, w_bench, annual_days=252):
    """Full-period total-risk and active-risk (tracking error) contributions at policy weights."""
    cols = list(asset_returns.columns)
    cov = asset_returns.cov(ddof=1).to_numpy() * annual_days
    wp = np.array([w_port.get(t, 0.0) for t in cols])
    wb = np.array([w_bench.get(t, 0.0) for t in cols])
    sigma, mctr, ctr = risk_contributions(cov, wp)
    te, amctr, actr = risk_contributions(cov, wp - wb)
    table = pd.DataFrame(
        {
            "weight": wp,
            "active_weight": wp - wb,
            "marginal_contribution": mctr,
            "risk_contribution": ctr,
            "pct_of_volatility": ctr / sigma,
            "active_risk_contribution": actr,
            "pct_of_tracking_error": actr / te,
        },
        index=pd.Index(cols, name="ticker"),
    )
    return sigma, te, table


def rolling_segment_risk(asset_returns, w_port, segments, window=252, annual_days=252):
    """At each month end: share of policy-weight volatility by segment, using the trailing window."""
    cols = list(asset_returns.columns)
    w = np.array([w_port.get(t, 0.0) for t in cols])
    month_end = asset_returns.groupby(asset_returns.index.to_period("M")).tail(1).index
    rows = {}
    for date in month_end:
        hist = asset_returns.loc[:date].tail(window)
        if len(hist) < window:
            continue
        sigma, _, ctr = risk_contributions(hist.cov(ddof=1).to_numpy() * annual_days, w)
        row = {"volatility": sigma}
        for s in dict.fromkeys(segments[t] for t in cols):
            row["pct_" + s] = float(sum(c for c, t in zip(ctr, cols) if segments[t] == s) / sigma)
        if "VTI" in cols and "BND" in cols:
            row["corr_VTI_BND"] = float(hist["VTI"].corr(hist["BND"]))
        rows[date] = row
    return pd.DataFrame(rows).T.rename_axis("date")


# ---------- VaR / ES forecasts ----------


def var_forecasts(asset_returns, weights_start, realized, levels=(0.95, 0.99), window=250, lam=0.94):
    """Rolling 1-day VaR and ES forecasts, reported as positive loss fractions.

    Methods
      historical : historical simulation, current weights applied to the trailing
                   `window` asset returns (empirical quantile, numpy 'linear')
      normal     : parametric normal, mean 0, covariance from the same window
      ewma       : parametric normal, RiskMetrics EWMA covariance (lambda=lam),
                   seeded with the first window's sample covariance
    The first forecast is for session index `window` (needs `window` prior days).
    """
    X = asset_returns.to_numpy()
    W = weights_start.reindex(columns=asset_returns.columns).to_numpy()
    n = len(X)
    if n <= window:
        raise ValueError("Not enough observations for the backtest window")
    nd = NormalDist()
    out = []
    ewma = np.cov(X[:window].T, ddof=1)
    for t in range(window, n):
        if t > window:
            r = X[t - 1][:, None]
            ewma = lam * ewma + (1 - lam) * (r @ r.T)
        hist = X[t - window : t]
        w = W[t]
        pnl = hist @ w
        cov = np.cov(hist.T, ddof=1)
        sig_n = math.sqrt(float(w @ cov @ w))
        sig_e = math.sqrt(float(w @ ewma @ w))
        row = {"date": asset_returns.index[t], "realized": float(realized.iloc[t])}
        for a in levels:
            tag = f"{int(round(a * 100))}"
            q = float(np.quantile(pnl, 1 - a))
            row[f"hist_var_{tag}"] = -q
            tail = pnl[pnl <= q]
            row[f"hist_es_{tag}"] = float(-tail.mean())
            z = nd.inv_cdf(a)
            es_mult = nd.pdf(z) / (1 - a)
            row[f"normal_var_{tag}"] = z * sig_n
            row[f"normal_es_{tag}"] = es_mult * sig_n
            row[f"ewma_var_{tag}"] = z * sig_e
            row[f"ewma_es_{tag}"] = es_mult * sig_e
        out.append(row)
    return pd.DataFrame(out).set_index("date")


# ---------- backtests ----------


def _xlogy(x, y):
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_pof(exceptions, n, p):
    """Kupiec (1995) proportion-of-failures likelihood ratio; chi-square(1) p-value."""
    x = int(exceptions)
    phat = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = max(0.0, -2 * (ll0 - ll1))
    return lr, math.erfc(math.sqrt(lr / 2))


def christoffersen(hits):
    """Christoffersen (1998) independence LR (chi-square(1)) using first-order transition counts."""
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr = max(0.0, -2 * (ll0 - ll1))
    return lr, math.erfc(math.sqrt(lr / 2)), {"n00": n00, "n01": n01, "n10": n10, "n11": n11}


def backtest(forecasts, levels=(0.95, 0.99), methods=("historical", "normal", "ewma")):
    """Exception counts and coverage/independence tests for every method and level."""
    rows = []
    n = len(forecasts)
    for m in methods:
        prefix = "hist" if m == "historical" else m
        for a in levels:
            tag = f"{int(round(a * 100))}"
            var = forecasts[f"{prefix}_var_{tag}"]
            es = forecasts[f"{prefix}_es_{tag}"]
            hits = (forecasts.realized < -var).astype(int)
            x = int(hits.sum())
            lr_pof, p_pof = kupiec_pof(x, n, 1 - a)
            lr_ind, p_ind, _ = christoffersen(hits)
            lr_cc = lr_pof + lr_ind
            loss_on_hit = -forecasts.realized[hits.astype(bool)]
            rows.append(
                {
                    "method": m,
                    "confidence": a,
                    "days": n,
                    "expected_exceptions": n * (1 - a),
                    "exceptions": x,
                    "exception_rate": x / n,
                    "kupiec_lr": lr_pof,
                    "kupiec_p": p_pof,
                    "independence_lr": lr_ind,
                    "independence_p": p_ind,
                    "conditional_coverage_lr": lr_cc,
                    "conditional_coverage_p": math.exp(-lr_cc / 2),
                    "avg_var": float(var.mean()),
                    "avg_es": float(es.mean()),
                    "avg_loss_on_exception": float(loss_on_hit.mean()) if x else float("nan"),
                    "avg_es_on_exception": float(es[hits.astype(bool)].mean()) if x else float("nan"),
                }
            )
    return pd.DataFrame(rows)
