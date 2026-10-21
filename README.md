# Portfolio Performance and Risk Analytics

[![tests](https://github.com/wangggggg88/portfolio_project/actions/workflows/tests.yml/badge.svg)](https://github.com/wangggggg88/portfolio_project/actions/workflows/tests.yml)

*Independent project · Python, pandas, NumPy, SQLite · eight USD-listed ETFs, 2021–2025* · [中文说明](README.zh-CN.md)

![Summary: performance, attribution, factor decomposition and volatility targeting](docs/summary.png)

## Goal

A diversified eight-ETF 60/40 portfolio returned **34.81%** over 2021–2025. A plain 60% VTI / 40% BND benchmark returned **44.68%**. The project asks three questions:

1. **How big is the gap**, once returns are measured correctly (drifting weights, monthly rebalancing, trading costs)?
2. **Why** did the portfolio fall behind: allocation, security selection, or factor exposure?
3. **What can be forecast and controlled**: return, or only risk?

## Data

- **Prices**: daily adjusted closes for the eight ETFs from Yahoo Finance (2020-12-31 to 2025-12-31), frozen in a checksummed snapshot.
- **Factors**: Fama-French five factors, momentum and the T-bill rate from the Kenneth R. French Data Library.
- **What is hypothetical**: only the portfolio weights. They are a study design, not a real account; there are no simulated prices or returns.

## Approach

| Step | Question | Method |
|---|---|---|
| 1. Measure | How large is the gap, and is it robust to costs? | Daily backtest with drifting weights, monthly rebalancing on the next session, 0 / 5 / 10 bps costs |
| 2. Attribute | Allocation or selection? | Brinson-Fachler by month, linked over five years with Carino factors |
| 3. Decompose risk | Where does risk sit? | Euler contributions to volatility and tracking error; VaR / ES with Kupiec and Christoffersen tests |
| 4. Explain | Is it skill or factor exposure? | Fama-French five factors plus momentum, daily, Newey-West standard errors |
| 5. Forecast | Can volatility be predicted out of sample? | Six models (63/252-day, EWMA, GARCH, GJR-GARCH, HAR), refit every 21 sessions, QLIKE and Diebold-Mariano tests |
| 6. Apply | Does the forecast help? | Filtered historical simulation VaR; a 10% volatility target without leverage |

**How the steps connect.** Both portfolios start every month at 60/40, so any gap must come from what is held inside each sleeve. Attribution shows where the gap sits (the small-cap and international equity sleeves). Factor regressions then test whether that is a lower market beta and a size tilt, or something left unexplained. Return direction turned out not to be forecastable, so the last part tests whether risk is, and whether that is useful.

## Key results

- **Gap**: −9.87 pp cumulative (34.81% vs 44.68%). Tracking error 2.27%, information ratio −0.65. At 10 bps costs the portfolio loses only a further 0.27 pp.
- **Attribution**: the allocation effect is exactly zero; the whole gap is selection. Equity sleeve −9.49 pp (VB −3.41, VEA −3.53, VWO −2.55); fixed income −0.39 pp.
- **Risk**: equities produce about 91% of the portfolio's volatility at a 60% weight. The VTI underweight drives about 86% of tracking error.
- **Factors**: the active return is −1.48% a year (annualized). About 55% of it comes from factor tilts, mainly a lower US market beta (0.55 vs 0.61) and a small-cap tilt. The remaining alpha (−0.67%, t = −0.9) is not significant.
- **Volatility forecasting**: models that react to recent volatility win. A 252-day window is significantly worse than EWMA in 16 of 20 tests. GJR-GARCH ranks first for the portfolio, but not significantly.
- **VaR**: 99% VaR passes the Kupiec test only when a responsive model is combined with filtered historical simulation (4 of 4 pass, e.g. EWMA 12 exceptions vs 10 expected). With a normal quantile, 2 of 4 fail; slow windows fail either way.
- **Volatility target** (10%, no leverage, monthly, 5 bps; settings fixed in advance): over 2022–2025, max drawdown falls from −21.4% to −17.2% and volatility from 11.2% to 9.5%. Annualized return is almost unchanged (5.10% vs 5.16%). The Sharpe difference is not significant.

Full reports: [attribution and risk](reports/v2/attribution_risk_report.pdf) · [factors](reports/factors/factor_report.pdf) · [volatility](reports/volatility/volatility_report.pdf)

## How the results are checked

- **38 unit tests**, covering:
  - hand-computed returns and rebalancing;
  - a no-look-ahead test, where changing future prices must not change past results;
  - a hand-worked Brinson example;
  - Kupiec by hand;
  - recovery of known factor betas and simulated GARCH parameters.
- **Independent re-computation**:
  - SQL (`LAG`) recomputes all 10,040 asset returns;
  - a units-based ledger recomputes all 12 NAV scenarios;
  - Excel formulas reproduce the attribution (recalculated in LibreOffice, not desktop Excel);
  - the hand-written optimizer matches scipy.
- **Reproducibility**: Windows and Linux runs give identical results, with pinned dependencies.
- **Data contract**: rerun on prices from the companion project, *Financial Data Pipeline and Quality Monitoring*, all 117 metrics match the snapshot within 3.1e-8.

## Limitations

- This is a retrospective study of one five-year window. Weights were fixed before prices were loaded, but the results are period-dependent and are not out-of-sample performance.
- Attribution has two segments, because the benchmark holds only VTI and BND.
- US factors are used for international and bond ETFs. The loadings describe exposures; they do not prove cause.
- Adjusted closes are a vendor total-return proxy. There are no taxes, slippage or cash ledger, and the risk-free rate is 0 in the Sharpe ratio.
- Market and factor data are not redistributed (see `.gitignore`). To rerun, fetch them first, as shown below.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests            # 38 tests
.\.venv\Scripts\python.exe -m src.fetch_data                                  # one-time price download -> data/snapshot
.\.venv\Scripts\python.exe -m src.run_analysis   --config config/full.json   # performance -> reports/full
.\.venv\Scripts\python.exe -m src.run_v2         --config config/full.json   # attribution, risk, VaR -> reports/v2
.\.venv\Scripts\python.exe scripts\fetch_factors.py                           # one-time factor download
.\.venv\Scripts\python.exe -m src.run_factors    --config config/full.json   # factors -> reports/factors
.\.venv\Scripts\python.exe -m src.run_volatility --config config/full.json   # volatility -> reports/volatility
.\.venv\Scripts\python.exe scripts\build_summary_figure.py                    # docs/summary.png
```

## Layout

```text
config/     portfolio weights, date range, costs
src/        portfolio.py · attribution.py · risk.py · factors.py · volatility.py · vol_target.py · run_*.py
tests/      38 unit tests
scripts/    data fetch, Excel check builders, pipeline contract check, summary figure
reports/    CSV, charts and PDF reports for each module
docs/       data dictionary, one-page summary, summary figure
```

## License

Code is released under the [MIT License](LICENSE). Market and factor data are not covered; they remain subject to the terms of Yahoo Finance and the Kenneth R. French Data Library.
