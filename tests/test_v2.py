import math
import unittest
import numpy as np
import pandas as pd
from src.portfolio import simulate, monthly_returns
from src.attribution import brinson_fachler, link, attribute, monthly_asset_returns
from src.risk import risk_contributions, kupiec_pof, christoffersen, var_forecasts, backtest


class AttributionTests(unittest.TestCase):
    def test_brinson_fachler_hand_example(self):
        wp = pd.Series({"E": .7, "F": .3}); wb = pd.Series({"E": .6, "F": .4})
        rp = pd.DataFrame({"E": [.10], "F": [.02]}); rb = pd.DataFrame({"E": [.08], "F": [.03]})
        Rp, Rb, eff = brinson_fachler(wp, wb, rp, rb)
        self.assertAlmostEqual(Rp.iloc[0], .076); self.assertAlmostEqual(Rb.iloc[0], .06)
        expected = {("allocation", "E"): .002, ("allocation", "F"): .003, ("selection", "E"): .012,
                    ("selection", "F"): -.004, ("interaction", "E"): .002, ("interaction", "F"): .001}
        for key, value in expected.items():
            self.assertAlmostEqual(eff[key].iloc[0], value)
        self.assertAlmostEqual(eff.iloc[0].sum(), .016)

    def test_carino_linking_sums_to_compounded_excess(self):
        Rp = pd.Series([.10, -.05]); Rb = pd.Series([.08, -.02])
        eff = pd.DataFrame({"a": [.01, -.01], "b": [.01, -.02]})
        linked, tp, tb = link(eff, Rp, Rb)
        self.assertAlmostEqual(tp, 1.10 * .95 - 1); self.assertAlmostEqual(tb, 1.08 * .98 - 1)
        self.assertAlmostEqual(linked.sum(), tp - tb, places=14)
        # hand Carino factors
        k = lambda a, b: (math.log1p(a) - math.log1p(b)) / (a - b)
        K = k(tp, tb)
        self.assertAlmostEqual(linked["a"], (.01 * k(.10, .08) - .01 * k(-.05, -.02)) / K, places=14)

    def prices(self):
        idx = pd.to_datetime(["2020-12-31", "2021-01-15", "2021-01-29", "2021-02-12", "2021-02-26", "2021-03-31"])
        return pd.DataFrame({"S1": [100, 104, 108, 105, 112, 115], "S2": [50, 51, 49, 52, 53, 52],
                             "B1": [20, 20.1, 20.3, 20.2, 20.0, 20.4], "B2": [10, 10, 10.1, 10.15, 10.1, 10.2]}, index=idx)

    def test_attribution_reconciles_with_simulation(self):
        p = self.prices()
        seg = {"S1": "Eq", "S2": "Eq", "B1": "Fi", "B2": "Fi"}
        port = {"S1": .3, "S2": .3, "B1": .2, "B2": .2}; bench = {"S1": .5, "B1": .5}
        a = attribute(p, port, bench, seg)
        sim = monthly_returns(simulate(p, port)["daily"])
        np.testing.assert_allclose(a["monthly_portfolio"].to_numpy(), sim.to_numpy(), atol=1e-14)
        self.assertAlmostEqual(a["by_effect"].sum(), a["total_portfolio"] - a["total_benchmark"], places=14)
        sel_int = a["by_effect"]["selection"] + a["by_effect"]["interaction"]
        self.assertAlmostEqual(a["assets"].selection_incl_interaction.sum(), sel_int, places=14)
        self.assertAlmostEqual(a["annual"].active.iloc[0], a["total_portfolio"] - a["total_benchmark"], places=14)

    def test_month_returns_use_anchor(self):
        r = monthly_asset_returns(self.prices())
        self.assertAlmostEqual(r.loc["2021-01", "S1"], .08)
        self.assertAlmostEqual(r.loc["2021-02", "S1"], 112 / 108 - 1)

    def test_segment_without_benchmark_weight_rejected(self):
        p = self.prices()
        with self.assertRaises(ValueError):
            attribute(p, {"S1": .5, "S2": .5}, {"B1": 1.0}, {"S1": "Eq", "S2": "Eq", "B1": "Fi", "B2": "Fi"})


class RiskTests(unittest.TestCase):
    def test_euler_contributions_hand(self):
        cov = np.diag([.04, .01]); w = [.5, .5]
        sigma, mctr, ctr = risk_contributions(cov, w)
        self.assertAlmostEqual(sigma, math.sqrt(.0125))
        self.assertAlmostEqual(ctr[0], .01 / math.sqrt(.0125))
        self.assertAlmostEqual(ctr.sum(), sigma)

    def test_kupiec_hand_and_zero_when_exact(self):
        lr, p = kupiec_pof(5, 250, .01)
        ph = 5 / 250
        hand = -2 * (245 * math.log(.99) + 5 * math.log(.01) - 245 * math.log(1 - ph) - 5 * math.log(ph))
        self.assertAlmostEqual(lr, hand); self.assertTrue(0 < p < 1)
        self.assertAlmostEqual(kupiec_pof(10, 1000, .01)[0], 0)

    def test_christoffersen_clustering_detected(self):
        spread = [1 if i % 20 == 0 else 0 for i in range(400)]
        clustered = [1 if 100 <= i < 120 else 0 for i in range(400)]
        self.assertLess(christoffersen(spread)[0], christoffersen(clustered)[0])
        self.assertLess(christoffersen(clustered)[1], .01)

    def synthetic(self, n=320, seed=7):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2021-01-01", periods=n)
        X = pd.DataFrame(rng.normal(0, .01, (n, 2)), index=idx, columns=["A", "B"])
        W = pd.DataFrame(.5, index=idx, columns=["A", "B"])
        return X, W, X @ np.array([.5, .5])

    def test_var_no_lookahead(self):
        X, W, real = self.synthetic()
        f1 = var_forecasts(X, W, real, window=250)
        X2 = X.copy(); X2.iloc[-1] = -0.5
        f2 = var_forecasts(X2, W, X2 @ np.array([.5, .5]), window=250)
        cols = [c for c in f1.columns if c != "realized"]
        pd.testing.assert_frame_equal(f1[cols], f2[cols])

    def test_historical_var_matches_window_quantile(self):
        X, W, real = self.synthetic()
        f = var_forecasts(X, W, real, window=250)
        pnl = X.iloc[0:250].to_numpy() @ np.array([.5, .5])
        self.assertAlmostEqual(f.hist_var_99.iloc[0], -np.quantile(pnl, .01))
        bt = backtest(f)
        self.assertEqual(len(bt), 6); self.assertTrue((bt.days == len(f)).all())


if __name__ == "__main__":
    unittest.main()
