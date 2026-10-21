# Data and Calculation Definitions

This is a personal retrospective simulation study. It uses no client holdings, real trades or internal company data.

## Data sources and securities

Data: the Yahoo Finance chart HTTPS endpoint. Adjusted close is defined at https://in.help.yahoo.com/kb/adjusted-close-sln28256.html. The vendor states that adjusted closes include dividend and split adjustments; this project uses them as a total-return proxy and never treats them as traded prices.

Use: personal study and research. Yahoo data are not part of this project's open license, and no right to redistribute market data is claimed. Terms: https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html. Raw data and local market samples are listed in `.gitignore` and are not published. Code and market data are kept separate.

ETF definitions follow the issuer pages: https://advisors.vanguard.com/investments/etfs and https://workplace.vanguard.com/fund-list/; IEF https://www.ishares.com/us/products/239456/ishares-710-year-treasury-bond-etf; TIP https://www.ishares.com/us/products/239467/ishares-tips-bond-etf.

| Ticker | Exposure | Notes |
|---|---|---|
| VTI | US total stock market | Core equity holding; also used in the benchmark |
| VB | US small-cap equity | Overlaps with VTI; represents a small-cap tilt, not a separate non-overlapping asset |
| VEA | International developed equity | USD-listed, but the underlying currency risk remains |
| VWO | Emerging markets equity | USD-listed, but the underlying currency risk remains |
| BND | US investment-grade aggregate bonds | Core bond holding; also used in the benchmark |
| IEF | US Treasuries, 7–10 years | Partly overlaps with BND |
| TIP | US inflation-protected Treasuries | Still exposed to price, duration and interest-rate changes |
| BNDX | International bonds | Differences from the other bond ETFs depend on the product definitions |

## Input files

`prices.csv`: one row per date and ticker; all prices in USD.

| Field | Type | Rule |
|---|---|---|
| date | YYYY-MM-DD | Trading date; ascending; must cover the frozen reference calendar |
| ticker | string | Must be one of the eight ETFs in the configuration |
| adj_close | float | Positive and finite; the only price used to compute returns |
| close | float | Vendor's unadjusted close, kept as delivered; not used for simulated trade amounts; may include split adjustments |
| price_type | string | Must be `adjusted_close` |
| source | string | `Yahoo Finance`; must not be empty |

`calendar.csv`: the frozen set of dates observed for VTI. It is used to detect missing dates in the other seven ETFs, rather than intersecting all ETFs, so that missing dates are never dropped silently. Weekends are not in the calendar. Limitation: this is not an independent exchange calendar, so a date missing for all eight ETFs at once cannot be detected.

`manifest.json`: download time (UTC), request URLs, SHA-256 checksums of the raw JSON and normalized CSV files, raw row counts and time zones. Raw responses are written to a new snapshot directory and never overwritten. A data update requires a new directory, new configuration dates and a fresh validation; the analysis itself never goes online.

## Portfolio method

The 2020-12-31 close is the price anchor, with NAV = 1. Returns start on the first trading day of 2021 and end on 2025-12-31, giving 1,255 return observations. The main, benchmark and alternative weights were written into the configuration before any data were downloaded. The study is a retrospective simulation: the weights were not actually proposed or traded in 2020.

Main portfolio, 60% equity / 40% bonds: 40% US equity and 20% international equity; the bond sleeve covers US aggregate bonds, Treasuries, TIPS and international bonds. The weights are meant to compare exposures and were not optimized on in-sample returns. The 60% VTI / 40% BND benchmark is an easy-to-read US stock-bond reference, not the only correct benchmark for global markets.

Daily asset return r(i,t) = P(i,t)/P(i,t-1) - 1. Daily gross portfolio return = sum(w(i,t-1) * r(i,t)). Between rebalances, weights drift with asset returns. Target weights are set at the month-end close and apply from the next session's close-to-close return; weights never depend on future prices.

Cost = rate * sum(abs(target weight - pre-trade drifted weight)). This is buys plus sells divided by pre-trade NAV, not divided by two. The initial purchase counts as 1. Net return = (1 - cost) * (1 + gross return) - 1. Costs shrink all holdings in proportion; cash, share quantities, taxes and slippage are not modeled. There is no rebalance and no liquidation cost on the final day, because no later return follows it.

Annual return = (final NAV / initial NAV)^(252 / number of returns) - 1. Standard deviations use ddof = 1. Annual volatility = standard deviation of daily returns * sqrt(252). Sharpe = mean daily excess return * 252 / (standard deviation of excess returns * sqrt(252)), with a zero risk-free rate as a simplifying assumption. A non-zero annual rate would be converted as (1 + rf)^(1/252) - 1.

Tracking error = std(daily net portfolio return - daily net benchmark return at the same cost) * sqrt(252). Information ratio = mean daily active return * 252 / tracking error. When a denominator is close to zero the result is undefined and shown as n.a. in the CSV files, never as 0.

Maximum drawdown = min(NAV / prior peak NAV - 1), where the prior peak includes the initial NAV of 1, so a loss on the first day counts. Monthly and annual returns compound daily returns as the product of (1 + r) minus 1; they are never summed. In the short sample the annual rows cover partial years and are not full-year results.

Two definitions of cumulative excess return are reported: `cumulative_excess_pp` is the difference between the two cumulative returns (percentage points), and `relative_wealth_return` is the ratio of final NAVs minus 1. They are not mixed.

Daily contribution = sum(previous weight * asset return) = gross portfolio return. Multi-period contributions are linked with previous-day NAV: sum(NAV(t-1) * daily contribution), for the zero-cost model only, so the cumulative wealth contributions of all assets add up to the total return. `active_wealth_reconciliation.csv` is the difference between the two portfolios' linked contributions and adds up to the cumulative return gap. It is not Brinson attribution: it does not separate allocation and selection effects and does not prove cause.

## Input checks

Duplicates, missing values, unknown securities, invalid dates, non-positive or empty prices, a wrong price type and a missing source all stop the analysis with the location of the problem; nothing is filled with zero. A daily price change above 15% in absolute value raises a warning and is kept. No warnings were raised on the full sample.

All key floating-point consistency checks use an absolute tolerance of 1e-10. No random numbers are used, so no random seed is needed.
