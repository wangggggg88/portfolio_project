import math
import unittest

import numpy as np
import pandas as pd

from src.vol_target import monthly_exposure, managed_returns, stats, memmel_test


class VolTargetTests(unittest.TestCase):
    def setUp(self):
        self.idx = ["2022-01-03", "2022-01-04", "2022-02-01", "2022-02-02"]
        self.r = pd.Series([0.01, -0.02, 0.03, 0.0], index=self.idx)
        self.rf = pd.Series(0.0001, index=self.idx)

    def test_exposure_formula_and_cap(self):
        f21 = pd.Series([(0.20 ** 2) * 21 / 252, (0.05 ** 2) * 21 / 252], index=["2022-01-03", "2022-02-01"])
        w = monthly_exposure(f21, ["2022-01-03", "2022-02-01"], 0.10, cap=1.0)
        self.assertAlmostEqual(w.iloc[0], 0.5)      # 10% / 20%
        self.assertAlmostEqual(w.iloc[1], 1.0)      # 10% / 5% = 2, capped at 1

    def test_managed_returns_and_costs_by_hand(self):
        w = pd.Series([0.5, 0.8], index=["2022-01-03", "2022-02-01"])
        out, wd, cost = managed_returns(self.r, self.rf, w, cost_bps=10)
        self.assertAlmostEqual(cost.iloc[0], 0.5 * 0.001)        # from 100% to 50%
        self.assertAlmostEqual(cost.iloc[2], 0.3 * 0.001)        # from 50% to 80%
        self.assertEqual(cost.iloc[1], 0.0)
        self.assertAlmostEqual(out.iloc[1], 0.5 * -0.02 + 0.5 * 0.0001)
        self.assertAlmostEqual(out.iloc[2], 0.8 * 0.03 + 0.2 * 0.0001 - 0.0003)

    def test_missing_month_rejected(self):
        with self.assertRaises(ValueError):
            managed_returns(self.r, self.rf, pd.Series([0.5], index=["2022-01-03"]))

    def test_full_exposure_equals_original(self):
        w = pd.Series([1.0, 1.0], index=["2022-01-03", "2022-02-01"])
        out, _, _ = managed_returns(self.r, self.rf, w)
        pd.testing.assert_series_equal(out, self.r, check_names=False)

    def test_memmel_symmetry_and_identity(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.001, 0.01, 1000); b = rng.normal(0.0, 0.01, 1000)
        z1, p1 = memmel_test(a, b); z2, p2 = memmel_test(b, a)
        self.assertAlmostEqual(z1, -z2); self.assertAlmostEqual(p1, p2)
        s = stats(pd.Series(a, index=pd.bdate_range("2022-01-03", periods=1000).strftime("%Y-%m-%d")), pd.Series(0.0, index=pd.bdate_range("2022-01-03", periods=1000).strftime("%Y-%m-%d")))
        self.assertAlmostEqual(s["sharpe"], a.mean() / a.std(ddof=1) * math.sqrt(252))


if __name__ == "__main__":
    unittest.main()
