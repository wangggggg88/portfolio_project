import unittest
import numpy as np
import pandas as pd
from src.portfolio import simulate, metrics, linked_contributions, monthly_returns
from src.quality import validate


class PortfolioTests(unittest.TestCase):
    def prices(self):
        return pd.DataFrame(
            {"A": [100, 110, 110, 121], "B": [100, 100, 110, 110]},
            index=pd.to_datetime(["2021-01-28", "2021-01-29", "2021-02-01", "2021-02-02"]),
        )

    def test_hand_computed_month_boundary(self):
        r = simulate(self.prices(), {"A": 0.5, "B": 0.5})
        # Jan29: 5%; reset at Jan29 close; Feb1: 5%; Feb2: 4.7619%.
        np.testing.assert_allclose(r["daily"].net_return, [0.05, 0.05, 0.1 * (0.5 / 1.05)], atol=1e-12)
        np.testing.assert_allclose(r["daily"].nav, [1.05, 1.1025, 1.155], atol=1e-12)
        np.testing.assert_allclose(r["weights_start"].iloc[1], [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(r["weights_start"].iloc[2], [0.5 / 1.05, 0.55 / 1.05], atol=1e-12)
        self.assertEqual(str(r["rebalances"].iloc[0].formation_date.date()), "2021-01-29")
        self.assertEqual(str(r["rebalances"].iloc[0].effective_date.date()), "2021-02-01")

    def test_not_daily_fixed_weight(self):
        p = self.prices()
        p.index = pd.to_datetime(["2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28"])
        r = simulate(p, {"A": 0.5, "B": 0.5})
        self.assertAlmostEqual(r["daily"].net_return.iloc[1], 0.1 * (0.5 / 1.05))
        self.assertNotAlmostEqual(r["daily"].net_return.iloc[1], 0.05)

    def test_no_lookahead_future_price_perturbation(self):
        a = self.prices()
        b = a.copy()
        b.iloc[-1, 0] = 999
        x = simulate(a, {"A": 0.5, "B": 0.5})
        y = simulate(b, {"A": 0.5, "B": 0.5})
        np.testing.assert_allclose(x["daily"].iloc[:-1], y["daily"].iloc[:-1])
        np.testing.assert_allclose(x["weights_start"], y["weights_start"])

    def test_cost_buys_plus_sells_initial_and_timing(self):
        r = simulate(self.prices(), {"A": 0.5, "B": 0.5}, 10)
        turn = abs(0.5 - 0.55 / 1.05) + abs(0.5 - 0.5 / 1.05)
        self.assertAlmostEqual(r["daily"].net_return.iloc[0], 0.999 * 1.05 - 1)
        self.assertAlmostEqual(r["daily"].net_return.iloc[1], (1 - turn * 0.001) * 1.05 - 1)
        self.assertEqual(r["daily"].cost_fraction.iloc[-1], 0)
        self.assertLess(
            r["daily"].nav.iloc[-1], simulate(self.prices(), {"A": 0.5, "B": 0.5})["daily"].nav.iloc[-1]
        )

    def test_daily_and_linked_contribution(self):
        r = simulate(self.prices(), {"A": 0.5, "B": 0.5})
        np.testing.assert_allclose(r["contributions"].sum(axis=1), r["daily"].gross_return)
        self.assertAlmostEqual(linked_contributions(r).sum(), 0.155)
        self.assertAlmostEqual(monthly_returns(r["daily"]).iloc[1], 0.1)

    def test_initial_loss_drawdown_and_zero_denominator(self):
        p = pd.DataFrame({"A": [100, 90, 90]}, index=pd.date_range("2021-01-01", periods=3))
        r = simulate(p, {"A": 1})["daily"]
        self.assertAlmostEqual(r.drawdown.min(), -0.1)
        m = metrics(r, r)
        self.assertTrue(np.isnan(m["information_ratio"]))
        self.assertEqual(m["tracking_error"], 0)
        p["A"] = 100
        r = simulate(p, {"A": 1})["daily"]
        self.assertTrue(np.isnan(metrics(r, r)["sharpe"]))

    def test_risk_metrics_independent_expected(self):
        d = simulate(self.prices(), {"A": 0.5, "B": 0.5})["daily"]
        b = simulate(self.prices(), {"B": 1})["daily"]
        m = metrics(d, b)
        self.assertAlmostEqual(m["annual_return"], 1.155 ** (252 / 3) - 1)
        self.assertAlmostEqual(
            m["annual_volatility"], np.std([0.05, 0.05, 0.1 * 0.5 / 1.05], ddof=1) * np.sqrt(252)
        )
        self.assertAlmostEqual(
            m["tracking_error"], np.std(d.net_return - b.net_return, ddof=1) * np.sqrt(252)
        )

    def test_invalid_weights_and_returns(self):
        with self.assertRaises(ValueError):
            simulate(self.prices(), {"A": 0.3, "B": 0.3})
        with self.assertRaises(ValueError):
            simulate(self.prices(), {"A": 1.1, "B": -0.1})
        p = self.prices()
        p.iloc[1, 0] = np.nan
        with self.assertRaises(ValueError):
            simulate(p, {"A": 0.5, "B": 0.5})


class QualityTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "date": ["2021-01-29", "2021-02-01", "2021-02-02"],
                "ticker": ["A"] * 3,
                "adj_close": [100.0, 110.0, 120.0],
                "price_type": ["adjusted_close"] * 3,
                "source": ["test"] * 3,
            }
        )

    def check(self, f):
        return validate(f, self.frame().date, ["A"], "2021-01-29", "2021-02-02")

    def test_clean_weekend_not_missing(self):
        p, issues = self.check(self.frame())
        self.assertIsNotNone(p)
        self.assertEqual(len(issues), 0)

    def test_duplicate_missing_negative_unknown_wrong_type(self):
        for rule in ["duplicate", "missing", "price", "ticker", "type"]:
            with self.subTest(rule=rule):
                f = self.frame()
                if rule == "duplicate":
                    f = pd.concat([f, f.iloc[[0]]])
                if rule == "missing":
                    f = f.iloc[:-1]
                if rule == "price":
                    f.loc[0, "adj_close"] = -1
                if rule == "ticker":
                    f.loc[0, "ticker"] = "UNKNOWN"
                if rule == "type":
                    f.loc[0, "price_type"] = "close"
                p, issues = self.check(f)
                self.assertIsNone(p)
                self.assertIn(rule, list(issues.rule))

    def test_jump_warned_not_deleted(self):
        f = self.frame()
        f.loc[1, "adj_close"] = 150
        p, issues = self.check(f)
        self.assertIsNotNone(p)
        self.assertEqual(p.iloc[1, 0], 150)
        self.assertTrue(issues.severity.eq("warning").all())


if __name__ == "__main__":
    unittest.main()
