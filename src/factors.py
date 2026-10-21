"""Fama-French factor regressions (returns-based style analysis).

Data: Kenneth R. French Data Library daily files, saved unmodified under
data/factors/raw/ (see data/factors/manifest.json). Values in the files are
percent per day and are converted to decimals here.

Models (daily, OLS with intercept, Newey-West HAC standard errors):
  FF3     : Mkt-RF, SMB, HML
  FF5     : + RMW, CMA
  FF5+MOM : + Mom
Dependent variables: portfolio excess returns (net return - RF) for each
configuration and each ETF, plus the active return Main - Benchmark (no RF,
because RF cancels). Alphas are annualised with 252 sessions.

Return decomposition: with an intercept, OLS residuals average exactly zero, so
mean(y) = alpha + sum_k beta_k * mean(f_k). Annualised (x252) this splits the
average return into factor contributions and alpha with no remainder.
"""

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+MOM": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}


def parse_french(path):
    """Parse a French Data Library daily CSV (inside a .zip or plain). Returns decimals indexed by ISO date."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            text = z.read(name).decode("latin-1")
    else:
        text = path.read_text(encoding="latin-1")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",") and any(c.isalpha() for c in l))
    header = [h.strip() for h in lines[start].split(",")]
    rows = []
    for l in lines[start + 1 :]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break  # end of the daily table (annual tables / copyright follow)
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    df["date"] = pd.to_datetime(df.date, format="%Y%m%d").dt.strftime("%Y-%m-%d")
    df = df.set_index("date").astype(float) / 100.0
    df.columns = [c.strip() for c in df.columns]
    if "Mom" not in df.columns and "Mom   " in df.columns:
        df = df.rename(columns={"Mom   ": "Mom"})
    return df


def ols_hac(y, X, lags=5):
    """OLS with intercept; Newey-West (Bartlett) HAC covariance. Returns dict of arrays."""
    y = np.asarray(y, float)
    Xc = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    n, k = Xc.shape
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    beta = XtX_inv @ Xc.T @ y
    e = y - Xc @ beta
    u = Xc * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))
    ss_res = float(e @ e)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    return {
        "coef": beta,
        "se": se,
        "t": beta / se,
        "resid": e,
        "r2": r2,
        "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k),
        "n": n,
    }


def regress_table(series, factors, annual_days=252, lags=5, excess=True):
    """Run every model for every series. series: dict name -> daily returns (date-indexed)."""
    rows = []
    for name, r in series.items():
        df = pd.concat([r.rename("r"), factors], axis=1, join="inner").dropna()
        y = df.r - df.RF if excess else df.r
        for model, cols in MODELS.items():
            res = ols_hac(y, df[cols], lags)
            row = {
                "series": name,
                "model": model,
                "observations": res["n"],
                "r2": res["r2"],
                "adj_r2": res["adj_r2"],
                "alpha_annual": res["coef"][0] * annual_days,
                "alpha_t": res["t"][0],
            }
            for i, c in enumerate(cols, start=1):
                row[f"beta_{c}"] = res["coef"][i]
                row[f"t_{c}"] = res["t"][i]
            rows.append(row)
    return pd.DataFrame(rows)


def decompose(r, factors, cols, annual_days=252, lags=5, excess=False):
    """Annualised split of the average return into factor contributions and alpha (sums exactly)."""
    df = pd.concat([r.rename("r"), factors], axis=1, join="inner").dropna()
    y = df.r - df.RF if excess else df.r
    res = ols_hac(y, df[cols], lags)
    parts = {"alpha": res["coef"][0] * annual_days}
    for i, c in enumerate(cols, start=1):
        parts[c] = res["coef"][i] * df[c].mean() * annual_days
    total = float(y.mean() * annual_days)
    out = pd.DataFrame({"component": list(parts), "annualised_contribution": list(parts.values())})
    out.loc[len(out)] = ["total (mean return x 252)", total]
    return out, abs(sum(parts.values()) - total)


def rolling_beta(r, factors, window=252, excess=True):
    """Trailing one-factor market beta (cov/var) using only data up to each date."""
    df = pd.concat([r.rename("r"), factors], axis=1, join="inner").dropna()
    y = df.r - df.RF if excess else df.r
    m = df["Mkt-RF"]
    return (y.rolling(window).cov(m) / m.rolling(window).var()).dropna()
