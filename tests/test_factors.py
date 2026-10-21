import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.factors import parse_french, ols_hac, regress_table, decompose, rolling_beta

SAMPLE = """This file was created using the 202608 CRSP database.
The 1-month TBill return is from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.00,   -0.34,   -0.01,    0.15,    0.012
19630702,    0.79,   -0.27,    0.27,   -0.07,   -0.19,    0.012

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


class FactorTests(unittest.TestCase):
    def test_parse_zip_percent_to_decimal(self):
        d = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(d / "f.zip", "w") as z:
            z.writestr("F-F_Research_Data_5_Factors_2x3_daily.CSV", SAMPLE)
        f = parse_french(d / "f.zip")
        self.assertEqual(list(f.index), ["1963-07-01", "1963-07-02"])
        self.assertAlmostEqual(f.loc["1963-07-01", "Mkt-RF"], -0.0067)
        self.assertAlmostEqual(f.loc["1963-07-02", "RF"], 0.00012)
        self.assertEqual(list(f.columns), ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

    def synthetic(self, n=1500, seed=3):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2021-01-01", periods=n).strftime("%Y-%m-%d")
        F = pd.DataFrame(
            rng.normal(0, 0.01, (n, 6)), index=idx, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
        )
        F["RF"] = 0.0001
        betas = np.array([0.9, 0.3, -0.2, 0.1, 0.05, -0.1])
        r = (
            0.0002
            + F[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].to_numpy() @ betas
            + F.RF
            + rng.normal(0, 0.002, n)
        )
        return pd.Series(r, index=idx), F, betas

    def test_recovers_known_betas_and_matches_lstsq(self):
        r, F, betas = self.synthetic()
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
        res = ols_hac(r - F.RF, F[cols])
        np.testing.assert_allclose(res["coef"][1:], betas, atol=0.02)
        X = np.column_stack([np.ones(len(r)), F[cols]])
        ref = np.linalg.lstsq(X, (r - F.RF).to_numpy(), rcond=None)[0]
        np.testing.assert_allclose(res["coef"], ref, atol=1e-12)

    def test_hac_lag0_equals_white(self):
        r, F, _ = self.synthetic(400)
        X = F[["Mkt-RF"]]
        res = ols_hac(r, X, lags=0)
        Xc = np.column_stack([np.ones(len(r)), X])
        e = res["resid"]
        inv = np.linalg.inv(Xc.T @ Xc)
        white = inv @ (Xc.T * e**2) @ Xc @ inv * len(r) / (len(r) - 2)
        np.testing.assert_allclose(res["se"], np.sqrt(np.diag(white)), rtol=1e-10)

    def test_decomposition_sums_exactly(self):
        r, F, _ = self.synthetic()
        out, err = decompose(r, F, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"], excess=True)
        self.assertLess(err, 1e-12)

    def test_table_and_rolling_beta_use_past_only(self):
        r, F, _ = self.synthetic(600)
        t = regress_table({"x": r}, F)
        self.assertEqual(set(t.model), {"FF3", "FF5", "FF5+MOM"})
        b1 = rolling_beta(r, F)
        r2 = r.copy()
        r2.iloc[-1] = 0.5
        b2 = rolling_beta(r2, F)
        pd.testing.assert_series_equal(b1.iloc[:-1], b2.iloc[:-1])


if __name__ == "__main__":
    unittest.main()
