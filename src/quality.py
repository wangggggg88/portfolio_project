"""Project-one input gate, not the full project-two monitoring pipeline."""

import numpy as np
import pandas as pd


def validate(frame, calendar, tickers, anchor, end):
    issues = []
    required = {"date", "ticker", "adj_close", "price_type", "source"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    f = frame.copy()
    f["date"] = pd.to_datetime(f.date, errors="coerce")
    f["adj_close"] = pd.to_numeric(f.adj_close, errors="coerce")

    def add(rule, severity, key, reason):
        issues.append({"rule": rule, "severity": severity, "record_key": str(key), "reason": reason})

    for i, row in f.iterrows():
        key = f"{row['ticker']}:{row['date']}"
        if pd.isna(row.date):
            add("date", "error", i, "Invalid date")
        if row.ticker not in tickers:
            add("ticker", "error", key, "Unknown ticker")
        if not np.isfinite(row.adj_close) or row.adj_close <= 0:
            add("price", "error", key, "Missing or non-positive adjusted price")
        if row.price_type != "adjusted_close":
            add("type", "error", key, "Wrong price type")
        if pd.isna(row.source):
            add("source", "error", key, "Missing source")
    for _, row in f[f.duplicated(["date", "ticker"], keep=False)].iterrows():
        add("duplicate", "error", f"{row.ticker}:{row.date}", "Duplicate date/ticker")
    expected = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    expected = expected[(expected >= pd.Timestamp(anchor)) & (expected <= pd.Timestamp(end))]
    if not len(expected) or expected[0] != pd.Timestamp(anchor) or expected[-1] != pd.Timestamp(end):
        add("calendar", "error", "coverage", "Calendar does not cover requested anchor/end")
    for ticker in tickers:
        actual = pd.DatetimeIndex(f.loc[f.ticker.eq(ticker), "date"])
        for date in expected.difference(actual):
            add("missing", "error", f"{ticker}:{date.date()}", "Missing reference trading session")
        for date in actual.difference(expected):
            add("extra", "error", f"{ticker}:{date}", "Unexpected reference date")
    result = pd.DataFrame(issues, columns=["rule", "severity", "record_key", "reason"])
    if not result.empty and result.severity.eq("error").any():
        return None, result
    prices = f.pivot(index="date", columns="ticker", values="adj_close").reindex(
        index=expected, columns=tickers
    )
    jumps = prices.pct_change(fill_method=None).abs() > 0.15
    for date, ticker in zip(*np.where(jumps.to_numpy())):
        add(
            "jump",
            "warning",
            f"{tickers[ticker]}:{prices.index[date].date()}",
            "Absolute daily change >15%; retained for review",
        )
    return prices, pd.DataFrame(issues, columns=["rule", "severity", "record_key", "reason"])
