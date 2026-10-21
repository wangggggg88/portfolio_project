"""Volatility forecasting: historical, EWMA, GARCH(1,1), GJR-GARCH(1,1) and HAR, evaluated out of sample.

Conventions
  * Returns are scaled to percent inside the GARCH likelihood for numerical stability; forecasts are
    returned as variances of decimal returns.
  * Every forecast for session t uses returns up to t-1 only. GARCH-type parameters are re-estimated
    every `refit` sessions on an expanding window and held fixed in between.
  * Mean return is set to zero (standard for daily volatility forecasting); the variance proxy for a
    1-day horizon is r_t^2 and for an h-day horizon the sum of the next h squared returns.
  * No scipy dependency: Gaussian quasi-maximum likelihood is maximised with a small Nelder-Mead.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- optimiser
def nelder_mead(f, x0, step=0.5, tol=1e-8, max_iter=2000):
    """Minimise f with the Nelder-Mead simplex (standard coefficients)."""
    n = len(x0)
    pts = [np.asarray(x0, float)]
    for i in range(n):
        p = np.asarray(x0, float).copy()
        p[i] += step
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(max_iter):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol):
            break
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1]); fr = f(xr)
        if vals[0] <= fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
            continue
        if fr < vals[0]:
            xe = centroid + 2 * (centroid - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
            continue
        xc = centroid + 0.5 * (pts[-1] - centroid); fc = f(xc)
        if fc < vals[-1]:
            pts[-1], vals[-1] = xc, fc
            continue
        for i in range(1, len(pts)):
            pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
            vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


# ---------------------------------------------------------------- GARCH family
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def _params(z, gjr):
    """Unconstrained z -> (omega, alpha, gamma, beta) with omega > 0 and persistence < 1."""
    omega = math.exp(z[0])
    persist = 0.9999 * _sig(z[1])                     # alpha + gamma/2 + beta
    share_arch = _sig(z[2])                           # share of persistence from the ARCH terms
    arch = persist * share_arch
    beta = persist - arch
    if gjr:
        g_share = _sig(z[3])                          # part of the ARCH budget given to the asymmetry term
        gamma = 2 * arch * g_share
        alpha = arch - gamma / 2
    else:
        gamma, alpha = 0.0, arch
    return omega, alpha, gamma, beta


def filter_variance(r, omega, alpha, gamma, beta, sigma2_0):
    """Conditional variances: s2[t] is the variance of r[t] given r[:t]; s2 has len(r)+1 entries
    (the last one is the one-step-ahead forecast after the final observation)."""
    s2 = [sigma2_0]
    for x in r:
        prev = s2[-1]
        s2.append(omega + (alpha + (gamma if x < 0 else 0.0)) * x * x + beta * prev)
    return s2


def neg_loglik(z, r, gjr, sigma2_0):
    omega, alpha, gamma, beta = _params(z, gjr)
    s2 = filter_variance(r, omega, alpha, gamma, beta, sigma2_0)
    ll = 0.0
    for x, v in zip(r, s2[:-1]):
        if v <= 0:
            return 1e12
        ll += math.log(v) + x * x / v
    return 0.5 * ll


def fit_garch(r_pct, gjr=False, start=None):
    """Gaussian QMLE on percent returns. Returns dict with omega, alpha, gamma, beta, persistence, loglik."""
    r = [float(x) for x in r_pct]
    s0 = float(np.var(r))
    z0 = start if start is not None else ([math.log(0.05 * s0), 3.0, -1.5] + ([0.0] if gjr else []))
    f = lambda z: neg_loglik(z, r, gjr, s0)
    z, val = nelder_mead(f, z0)
    for _ in range(8):  # restart from the best point until the simplex stops finding improvements
        z2, val2 = nelder_mead(f, z, step=0.25, tol=1e-10)
        if val2 >= val - 1e-9:
            break
        z, val = z2, val2
    omega, alpha, gamma, beta = _params(z, gjr)
    return {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
            "persistence": alpha + gamma / 2 + beta, "neg_loglik": val, "z": list(z)}


def garch_term_variance(s2_next, p, h):
    """Sum of variances over the next h sessions, starting from the one-step forecast s2_next."""
    pers = p["persistence"]
    if pers >= 1:
        return h * s2_next
    vbar = p["omega"] / (1 - pers)
    k = np.arange(h)
    return float(np.sum(vbar + pers ** k * (s2_next - vbar)))


# ---------------------------------------------------------------- forecasts
def forecasts(r, start, horizons=(1, 21), refit=21, lam=0.94):
    """Out-of-sample variance forecasts for sessions start..len(r)-1.

    r: pd.Series of daily decimal returns. Returns {h: DataFrame(index=dates, columns=models)} and the
    GARCH parameter history. Forecast row t uses r[:t] only.
    """
    x = r.to_numpy(float)
    n = len(x)
    pct = x * 100.0
    out = {h: {} for h in horizons}
    models = ["hist63", "hist252", "ewma", "garch", "gjr", "har"]
    for h in horizons:
        out[h] = {m: np.full(n, np.nan) for m in models}
    history = []
    # Warm-up variances (sessions WARM..start-1) for standardizing past returns in filtered historical
    # simulation. They use only information available at the first forecast date `start`.
    WARM = 63
    warm = {m: np.full(n, np.nan) for m in models}
    for t in range(WARM, start):
        warm["hist63"][t] = float(np.mean(x[t - 63:t] ** 2))
        warm["hist252"][t] = float(np.mean(x[max(0, t - 252):t] ** 2))
    ew = float(np.mean(x[:WARM] ** 2))          # EWMA seeded on the first 63 returns, then run forward
    for t in range(WARM, start):
        if t > WARM:
            ew = lam * ew + (1 - lam) * x[t - 1] ** 2
        warm["ewma"][t] = ew
    ew = lam * ew + (1 - lam) * x[start - 1] ** 2
    ew_series = np.full(n + 1, np.nan)
    ew_series[start] = ew
    params = {}
    s2 = {}
    for t in range(start, n):
        if t > start:
            ew = lam * ew + (1 - lam) * x[t - 1] ** 2
        ew_series[t] = ew
        if (t - start) % refit == 0:
            for name, gjr in (("garch", False), ("gjr", True)):
                z_start = params.get(name, {}).get("z")
                p = fit_garch(pct[:t], gjr=gjr, start=z_start)
                params[name] = p
                s0 = float(np.var(pct[:t]))
                path = filter_variance(pct[:t], p["omega"], p["alpha"], p["gamma"], p["beta"], s0)
                s2[name] = path[-1]
                if t == start:
                    warm[name][WARM:start] = np.asarray(path[WARM:start]) / 1e4
                history.append({"date": r.index[t], "model": name, **{k: v for k, v in p.items() if k != "z"}})
        else:
            for name in ("garch", "gjr"):
                p = params[name]
                prev = x[t - 1] * 100.0
                s2[name] = p["omega"] + (p["alpha"] + (p["gamma"] if prev < 0 else 0.0)) * prev * prev + p["beta"] * s2[name]
        for h in horizons:
            out[h]["hist63"][t] = h * float(np.mean(x[t - 63:t] ** 2))
            out[h]["hist252"][t] = h * float(np.mean(x[max(0, t - 252):t] ** 2))
            out[h]["ewma"][t] = h * ew
            for name in ("garch", "gjr"):
                out[h][name][t] = garch_term_variance(s2[name], params[name], h) / 1e4
    # HAR on squared returns: target = sum of next h squared returns; regressors = daily, weekly, monthly means
    sq = pd.Series(x ** 2, index=r.index)
    d = sq.shift(0)
    w = sq.rolling(5).mean()
    m = sq.rolling(22).mean()
    for h in horizons:
        target = sum(sq.shift(-k) for k in range(h))          # target at row t: sum of r^2 over t..t+h-1
        X_all = np.column_stack([np.ones(n), d.shift(1), w.shift(1), m.shift(1)])  # info up to t-1
        har_beta = None
        for t in range(start, n):
            # training rows s with complete target known before t: s + h - 1 <= t - 1
            last = t - h
            rows = np.arange(23, last + 1)
            if len(rows) < 100:
                continue
            if (t - start) % refit == 0 or har_beta is None:
                Xa = X_all[rows]; ya = target.to_numpy()[rows]
                ok = np.isfinite(Xa).all(axis=1) & np.isfinite(ya)
                har_beta = np.linalg.lstsq(Xa[ok], ya[ok], rcond=None)[0]
            f = float(X_all[t] @ har_beta)
            out[h]["har"][t] = max(f, 0.05 * out[h]["hist252"][t])  # floor: HAR can go negative
            if h == 1 and t == start:
                for s_ in range(WARM, start):
                    warm["har"][s_] = max(float(X_all[s_] @ har_beta), 0.05 * warm["hist252"][s_])
    warm_df = pd.DataFrame(warm, index=r.index).iloc[WARM:start]
    frames = {}
    for h in horizons:
        df = pd.DataFrame(out[h], index=r.index).iloc[start:]
        realized = sum(sq.shift(-k) for k in range(h)).iloc[start:]
        df.insert(0, "realized", realized)
        frames[h] = df
    return frames, pd.DataFrame(history), warm_df


# ---------------------------------------------------------------- evaluation
def qlike(realized, forecast):
    """Patton's robust QLIKE loss: RV/F - log(RV/F) - 1 (0 for a perfect forecast). Zero RV is skipped."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = realized / forecast
    ok = (realized > 0) & np.isfinite(ratio)
    return ratio[ok] - np.log(ratio[ok]) - 1


def mse(realized, forecast):
    return (realized - forecast) ** 2


def newey_west_var(d, lags):
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    s = float(d @ d) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * float(d[l:] @ d[:-l]) / n
    return s


def diebold_mariano(loss_a, loss_b, lags):
    """DM test of equal predictive accuracy. Negative statistic: model a has lower loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    var = newey_west_var(d, lags) / len(d)
    stat = float(np.mean(d) / math.sqrt(var)) if var > 0 else float("nan")
    return stat, math.erfc(abs(stat) / math.sqrt(2))


def mincer_zarnowitz(realized, forecast):
    """Regress realized on forecast: unbiased forecasts have intercept 0 and slope 1."""
    X = np.column_stack([np.ones(len(forecast)), forecast])
    b = np.linalg.lstsq(X, realized, rcond=None)[0]
    resid = realized - X @ b
    r2 = 1 - float(resid @ resid) / float(((realized - realized.mean()) ** 2).sum())
    return float(b[0]), float(b[1]), r2


def evaluate(frame, h):
    """Loss table and DM tests (each model vs EWMA) on rows where the realized proxy is complete."""
    df = frame.dropna(subset=["realized"])
    models = [c for c in df.columns if c != "realized"]
    df = df.dropna(subset=models)
    rows, dm = [], []
    lags = max(h - 1, 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        ql = {m: np.asarray(df.realized / df[m] - np.log(df.realized / df[m]) - 1) for m in models}
    keep = np.all(np.isfinite(np.column_stack(list(ql.values()))), axis=1) & (df.realized.to_numpy() > 0)
    for m in models:
        a, b, r2 = mincer_zarnowitz(df.realized.to_numpy(), df[m].to_numpy())
        rows.append({"model": m, "horizon": h, "observations": int(keep.sum()), "qlike": float(np.mean(ql[m][keep])),
                     "mse_x1e8": float(np.mean(mse(df.realized, df[m]))) * 1e8, "mz_intercept_x1e4": a * 1e4,
                     "mz_slope": b, "mz_r2": r2,
                     "avg_forecast_vol_ann": float(np.mean(np.sqrt(df[m] / h * 252))),
                     "avg_realized_vol_ann": float(np.sqrt(np.mean(df.realized) / h * 252))})
        if m != "ewma":
            stat, p = diebold_mariano(ql[m][keep], ql["ewma"][keep], lags)
            dm.append({"model": m, "vs": "ewma", "horizon": h, "dm_stat_qlike": stat, "p_value": p})
    t = pd.DataFrame(rows)
    t["rank_qlike"] = t.qlike.rank().astype(int)
    return t, pd.DataFrame(dm)


# ---------------------------------------------------------------- VaR application
def var_from_vol(frame1, r, warm=None, levels=(0.95, 0.99), fhs_window=250):
    """1-day VaR from each model's variance forecast: normal quantile, and filtered historical simulation
    (empirical quantile of the past `fhs_window` standardized returns r/sigma, scaled by today's sigma).
    `warm` holds warm-up variances before the first forecast so FHS covers the same days as the normal VaR."""
    nd = NormalDist()
    rows = {}
    models = [c for c in frame1.columns if c != "realized"]
    var_all = pd.concat([warm[models], frame1[models]]) if warm is not None else frame1[models]
    x_all = r.reindex(var_all.index)
    for m in models:
        sig_all = np.sqrt(var_all[m])
        z = x_all / sig_all
        sig = sig_all.reindex(frame1.index)
        for a in levels:
            tag = int(round(a * 100))
            rows[f"{m}_normal_{tag}"] = nd.inv_cdf(a) * sig
            q = z.shift(1).rolling(fhs_window, min_periods=100).quantile(1 - a).reindex(frame1.index)  # past only
            rows[f"{m}_fhs_{tag}"] = -q * sig
    x = r.reindex(frame1.index)
    out = pd.DataFrame(rows, index=frame1.index)
    out.insert(0, "realized_return", x)
    return out
