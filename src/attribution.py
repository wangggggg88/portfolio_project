"""Brinson-Fachler performance attribution with Carino multi-period linking.

Period = calendar month. Because the model rebalances to target at every month
end and carries no cost in the zero-cost scenario, each month is a buy-and-hold
period that starts at target weights. The monthly portfolio return is therefore
exactly sum(w_i * R_i,month), which makes a monthly holdings-based attribution
exact rather than an approximation (verified in run_v2 against simulate()).

Only the zero-cost scenario is attributed; costs would appear as an unexplained
residual and are reported separately in cost_sensitivity.csv.
"""
import math
import numpy as np
import pandas as pd


def monthly_asset_returns(prices):
    """Month-end to month-end returns per asset; the first month starts at the anchor price."""
    month_end = prices.groupby(prices.index.to_period("M")).tail(1)
    returns = month_end.pct_change(fill_method=None).iloc[1:]
    returns.index = returns.index.to_period("M")
    return returns


def segment_view(weights, asset_returns, segments):
    """Aggregate asset weights/returns into segments.

    weights: Series of month-start weights by ticker.
    asset_returns: DataFrame (months x tickers).
    segments: dict ticker -> segment name.
    Returns (segment weights Series, segment returns DataFrame). A segment with
    zero weight has an undefined return (NaN).
    """
    names = list(dict.fromkeys(segments[t] for t in asset_returns.columns))
    w = pd.Series({s: sum(weights.get(t, 0.0) for t in asset_returns.columns if segments[t] == s) for s in names})
    r = pd.DataFrame(index=asset_returns.index, columns=names, dtype=float)
    for s in names:
        members = [t for t in asset_returns.columns if segments[t] == s]
        if w[s] > 0:
            r[s] = sum(weights.get(t, 0.0) * asset_returns[t] for t in members) / w[s]
        else:
            r[s] = np.nan
    return w, r


def brinson_fachler(wp, wb, rp, rb):
    """Single-period Brinson-Fachler effects per segment for every period.

    wp, wb: Series of segment weights (constant month-start weights).
    rp, rb: DataFrames (periods x segments) of segment returns.
    allocation  = (wp - wb) * (rb_s - Rb)
    selection   = wb * (rp_s - rb_s)
    interaction = (wp - wb) * (rp_s - rb_s)
    Every segment the portfolio holds must also be held by the benchmark,
    otherwise the benchmark segment return is undefined; such segments must be
    grouped into a broader segment first.
    """
    held = wp[wp > 0].index
    if (wb.reindex(held).fillna(0) <= 0).any():
        raise ValueError("Portfolio holds a segment with no benchmark weight; regroup segments")
    Rb = (rb.fillna(0) * wb).sum(axis=1)
    Rp = (rp.fillna(0) * wp).sum(axis=1)
    # Segments not held by the portfolio have no portfolio return: set it to the
    # benchmark segment return so selection/interaction are zero there.
    rp_f = rp.copy()
    for s in rp.columns:
        if wp.get(s, 0.0) <= 0:
            rp_f[s] = rb[s]
    alloc = (wp - wb) * rb.sub(Rb, axis=0)
    select = wb * (rp_f - rb)
    inter = (wp - wb) * (rp_f - rb)
    effects = pd.concat({"allocation": alloc, "selection": select, "interaction": inter}, axis=1)
    return Rp, Rb, effects


def carino_factors(Rp, Rb):
    """Carino (1999) smoothing: k_t = [ln(1+Rp_t) - ln(1+Rb_t)] / (Rp_t - Rb_t), K likewise for the total."""
    def k(a, b):
        return 1 / (1 + a) if abs(a - b) < 1e-15 else (math.log1p(a) - math.log1p(b)) / (a - b)
    kt = pd.Series([k(a, b) for a, b in zip(Rp, Rb)], index=Rp.index)
    total_p = float(np.prod(1 + Rp) - 1)
    total_b = float(np.prod(1 + Rb) - 1)
    return kt, k(total_p, total_b), total_p, total_b


def link(effects, Rp, Rb):
    """Link single-period effects so they sum exactly to cumulative Rp - Rb."""
    kt, K, total_p, total_b = carino_factors(Rp, Rb)
    linked = effects.mul(kt / K, axis=0).sum()
    return linked, total_p, total_b


def attribute(prices, port_w, bench_w, segments):
    """Full attribution of one portfolio against a benchmark.

    Returns dict with monthly effects, linked totals by segment and effect,
    an asset-level split of (selection + interaction), and annual linked totals.
    """
    R = monthly_asset_returns(prices)
    wp_s, rp_s = segment_view(port_w, R, segments)
    wb_s, rb_s = segment_view(bench_w, R, segments)
    Rp, Rb, eff = brinson_fachler(wp_s, wb_s, rp_s, rb_s)
    linked, total_p, total_b = link(eff, Rp, Rb)
    by_effect = linked.groupby(level=0).sum()
    by_segment = linked.unstack(level=0)[["allocation", "selection", "interaction"]]
    by_segment["total"] = by_segment.sum(axis=1)
    # Asset split of selection+interaction: wp_i * (R_i - rb_segment); sums exactly to the segment's sel+int.
    kt, K, _, _ = carino_factors(Rp, Rb)
    asset_rows = {}
    for t in R.columns:
        s = segments[t]
        w = port_w.get(t, 0.0)
        if w == 0:
            continue
        asset_rows[t] = {"segment": s, "weight": w,
                         "selection_incl_interaction": float((w * (R[t] - rb_s[s]) * kt / K).sum())}
    assets = pd.DataFrame(asset_rows).T
    assets["weight"] = assets.weight.astype(float)
    assets["selection_incl_interaction"] = assets.selection_incl_interaction.astype(float)
    annual = {}
    for year, idx in Rp.groupby(Rp.index.year).groups.items():
        l, tp, tb = link(eff.loc[idx], Rp.loc[idx], Rb.loc[idx])
        row = l.groupby(level=0).sum().to_dict()
        row.update({"portfolio": tp, "benchmark": tb, "active": tp - tb})
        annual[year] = row
    annual = pd.DataFrame(annual).T[["portfolio", "benchmark", "active", "allocation", "selection", "interaction"]]
    return {"monthly_portfolio": Rp, "monthly_benchmark": Rb, "monthly_effects": eff,
            "by_effect": by_effect, "by_segment": by_segment, "assets": assets, "annual": annual,
            "total_portfolio": total_p, "total_benchmark": total_b,
            "segment_weights": pd.DataFrame({"portfolio": wp_s, "benchmark": wb_s})}
