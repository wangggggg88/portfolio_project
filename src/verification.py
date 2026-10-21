"""Independent stdlib units ledger, SQL returns and statistics checks."""
import math
import sqlite3
import statistics
import numpy as np

SQL = '''WITH prior AS (
 SELECT date,ticker,adj_close,
 LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS previous
 FROM prices
) SELECT date,ticker,adj_close/previous-1 AS daily_return
FROM prior WHERE previous IS NOT NULL ORDER BY date,ticker;'''

def verify(prices, results, cfg, all_results):
    db = sqlite3.connect(":memory:")
    prices.rename_axis("date").reset_index().melt(id_vars="date", var_name="ticker", value_name="adj_close").assign(date=lambda x:x.date.dt.strftime("%Y-%m-%d")).to_sql("prices", db, index=False)
    python = prices.pct_change(fill_method=None)
    err_sql = max(abs(value-python.loc[date,ticker]) for date,ticker,value in db.execute(SQL))
    db.close()
    # Independent virtual units ledger, no use of simulate's weight recurrence.
    # Adjusted-price units represent reinvested total-return exposure, not shares.
    dates = list(prices.index)
    p = prices.to_numpy().tolist()
    target = [cfg["portfolios"]["Main"].get(t,0) for t in prices.columns]
    units = [w/v for w,v in zip(target,p[0])]
    navs = []
    for i in range(1,len(p)):
        value = sum(u*v for u,v in zip(units,p[i]))
        navs.append(value)
        if i+1 < len(p) and (dates[i].year,dates[i].month)!=(dates[i+1].year,dates[i+1].month):
            units = [value*w/v for w,v in zip(target,p[i])]
    main = results["Main"]["daily"]
    err_units = max(abs(a-b) for a,b in zip(navs,main.nav))
    r = main.net_return.tolist()
    err_vol = abs(statistics.stdev(r)*math.sqrt(cfg["annual_days"])-np.std(r,ddof=1)*np.sqrt(cfg["annual_days"]))
    errors = {"sql_price_returns_max_abs_error": err_sql, "independent_units_nav_max_abs_error": err_units,
              "stdlib_volatility_abs_error": err_vol,
              "daily_contribution_max_abs_error": float((results["Main"]["contributions"].sum(axis=1)-main.gross_return).abs().max()),
              "nav_return_max_abs_error": float((main.nav.pct_change().iloc[1:]-main.net_return.iloc[1:]).abs().max())}
    cost_error = 0.0
    for bps, scenarios in all_results.items():
        for name, scenario in scenarios.items():
            targets = [cfg["portfolios"][name].get(t,0) for t in prices.columns]
            holding_units = [w/v for w,v in zip(targets,p[0])]
            pending = 1.0
            for i in range(1,len(p)):
                holding_units = [u*(1-pending*bps/10000) for u in holding_units]
                value = sum(u*v for u,v in zip(holding_units,p[i]))
                cost_error = max(cost_error,abs(value-scenario["daily"].nav.iloc[i-1]))
                pending = 0.0
                if i+1<len(p) and (dates[i].year,dates[i].month)!=(dates[i+1].year,dates[i+1].month):
                    drift = [u*v/value for u,v in zip(holding_units,p[i])]
                    pending = sum(abs(a-b) for a,b in zip(targets,drift))
                    holding_units = [value*w/v for w,v in zip(targets,p[i])]
    errors["independent_units_all_cost_scenarios_nav_max_abs_error"] = cost_error
    tol = cfg["tolerance"]
    if not all(v <= tol for v in errors.values()): raise AssertionError(errors)
    return {"status":"passed", "absolute_tolerance":tol, "checks":errors,
            "independent_model":"Adjusted-price virtual units; not actual tradable shares", "sql_rows_checked": (len(prices)-1)*len(prices.columns),
            "cost_scenarios_checked":len(all_results)*len(results)}
