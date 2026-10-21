-- Purpose: independently calculate adjusted-price returns in SQLite.
-- Input: prices(date TEXT,ticker TEXT,adj_close REAL), unique date/ticker.
-- Output: date,ticker,daily_return, excluding the initial price anchor.
-- Sample: A at 100 then 110 returns 0.1. Actual full check: 10,040 rows.
WITH prior AS (
  SELECT date, ticker, adj_close,
         LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS previous
  FROM prices
)
SELECT date, ticker, adj_close / previous - 1 AS daily_return
FROM prior
WHERE previous IS NOT NULL
ORDER BY date, ticker;
