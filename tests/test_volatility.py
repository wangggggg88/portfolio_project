import math
import unittest

import numpy as np
import pandas as pd

from src.volatility import (
    nelder_mead,
    fit_garch,
    filter_variance,
    garch_term_variance,
    forecasts,
    qlike,
    diebold_mariano,
    mincer_zarnowitz,
    var_from_vol,
    neg_loglik,
)


def simulate_gjr(n, omega, alpha, gamma, beta, seed):
    rng = np.random.default_rng(seed)
    s2 = omega / (1 - alpha - gamma / 2 - beta)
    r = []
    for _ in range(n):
        x = rng.standard_normal() * math.sqrt(s2)
        r.append(x)
        s2 = omega + (alpha + (gamma if x < 0 else 0)) * x * x + beta * s2
    return np.array(r)


class VolatilityTests(unittest.TestCase):
    def test_nelder_mead_quadratic(self):
        x, f = nelder_mead(lambda z: (z[0] - 1) ** 2 + 3 * (z[1] + 2) ** 2, [0.0, 0.0], tol=1e-14)
        np.testing.assert_allclose(x, [1, -2], atol=1e-5)

    def test_recovers_gjr_parameters(self):
        r = simulate_gjr(4000, 0.02, 0.04, 0.10, 0.88, seed=11)
        p = fit_garch(r, gjr=True)
        self.assertAlmostEqual(p["beta"], 0.88, delta=0.04)
        self.assertAlmostEqual(p["gamma"], 0.10, delta=0.05)
        self.assertGreater(p["gamma"], p["alpha"])  # asymmetry detected
        self.assertLess(p["persistence"], 1)

    def test_fit_is_local_minimum(self):
        r = simulate_gjr(1500, 0.05, 0.08, 0.0, 0.88, seed=3)
        p = fit_garch(r, gjr=False)
        s0 = float(np.var(r))
        rng = np.random.default_rng(0)
        base = neg_loglik(p["z"], list(r), False, s0)
        for _ in range(50):
            self.assertGreaterEqual(
                neg_loglik(np.array(p["z"]) + rng.normal(0, 0.05, 3), list(r), False, s0), base - 1e-9
            )

    def test_filter_and_term_structure(self):
        s2 = filter_variance([1.0, -2.0], 0.1, 0.05, 0.1, 0.8, 1.0)
        self.assertAlmostEqual(s2[1], 0.1 + 0.05 * 1 + 0.8 * 1.0)
        self.assertAlmostEqual(s2[2], 0.1 + 0.15 * 4 + 0.8 * s2[1])
        p = {"omega": 0.1, "persistence": 0.9}
        self.assertAlmostEqual(garch_term_variance(2.0, p, 1), 2.0)
        vbar = 1.0
        self.assertAlmostEqual(garch_term_variance(2.0, p, 3), 3 * vbar + (2.0 - vbar) * (1 + 0.9 + 0.81))

    def test_qlike_and_mz_by_hand(self):
        self.assertAlmostEqual(float(qlike(np.array([2.0]), np.array([2.0]))[0]), 0.0)
        self.assertAlmostEqual(float(qlike(np.array([2.0]), np.array([1.0]))[0]), 2 - math.log(2) - 1)
        a, b, r2 = mincer_zarnowitz(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(a, 0)
        self.assertAlmostEqual(b, 1)
        self.assertAlmostEqual(r2, 1)

    def test_diebold_mariano_sign(self):
        rng = np.random.default_rng(1)
        a = rng.normal(1.0, 0.1, 500)
        b = a + 0.05
        stat, p = diebold_mariano(a, b, 5)
        self.assertLess(stat, 0)
        self.assertLess(p, 0.01)

    def test_forecasts_use_only_past(self):
        r = pd.Series(
            simulate_gjr(420, 2e-6, 0.05, 0.05, 0.9, seed=5),
            index=pd.bdate_range("2021-01-01", periods=420).strftime("%Y-%m-%d"),
        )
        f1, _, w1 = forecasts(r, 300)
        r2 = r.copy()
        r2.iloc[-1] = 0.2
        f2, _, w2 = forecasts(r2, 300)
        cols = [c for c in f1[1].columns if c != "realized"]
        pd.testing.assert_frame_equal(f1[1][cols], f2[1][cols])  # last return enters no forecast
        pd.testing.assert_frame_equal(w1, w2)
        v1 = var_from_vol(f1[1], r, w1)
        v2 = var_from_vol(f2[1], r2, w2)
        pd.testing.assert_frame_equal(v1.drop(columns="realized_return"), v2.drop(columns="realized_return"))


if __name__ == "__main__":
    unittest.main()
