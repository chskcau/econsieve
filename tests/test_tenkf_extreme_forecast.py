"""One archived forecast update, independent of its originating model and solver."""

from pathlib import Path

import numpy as np
import pytest
from mpmath import mp

from econsieve import ForecastPrecisionRequired, TEnKF


def reference_update(states, forecasts, observation, perturbations, covariance):
    """Condition the empirical joint Gaussian at 200 decimal digits."""
    with mp.workdps(200):
        x, y, noise = [mp.matrix(array.T.tolist()) for array in (states, forecasts, perturbations)]
        z, r = mp.matrix(observation.tolist()), mp.matrix(covariance.tolist())
        n = y.cols
        ones = mp.ones(n, 1)
        x_mean, y_mean = x * ones / n, y * ones / n
        xc, yc = x - x_mean * ones.T, y - y_mean * ones.T
        s = yc * yc.T / (n - 1) + r
        residual = z - y_mean
        log_density = -(y.rows * mp.log(2 * mp.pi) + mp.log(mp.det(s))
                        + (residual.T * mp.lu_solve(s, residual))[0]) / 2
        gain = (xc * yc.T / (n - 1)) * s**-1
        updated = x + gain * (z * ones.T - y - noise)
        return float(log_density), np.array(updated.tolist(), dtype=float)


@pytest.fixture(scope="module")
def extreme_forecast():
    # CapCons small NK, 2009Q4, step 60, walker 23, upstream filter draws.
    # Freeze the 32 members at the forecast boundary so solver or RNG changes
    # cannot remove this input. Member 30 has observations of order 1e17.
    rows = np.loadtxt(Path(__file__).with_name("tenkf_2009q4.txt"))
    states, forecasts, perturbations = np.split(rows, [5, 8], axis=1)
    observation = np.array([0.8018662103571383, 0.31588117555321205, 0.03])
    covariance = np.eye(3) * 1e-12
    members = iter(zip(states, forecasts))
    filt = TEnKF(len(states), states.shape[1], forecasts.shape[1], fx=lambda state, eps: (next(members), 0))
    filt.R = covariance

    def draws(mean, cov, size, **kwargs):
        if cov is filt.R:
            return perturbations[None]
        assert cov is filt.Q
        return np.zeros((*size, len(mean)))

    filt.multivariate = draws
    expected_ll, expected_states = reference_update(states, forecasts, observation, perturbations, covariance)
    parts = np.empty(1)
    ll = filt.batch_filter(observation[None], init_states=states.T, calc_ll=True, store=True,
                           ll_contributions=parts)
    assert parts[0] == ll == filt.ll
    assert filt.fallback_decimal_digits == 200
    assert filt.precision_requests == filt.high_precision_updates == filt.high_precision_completed == 1
    assert filt.high_precision_seconds > 0
    return ll, expected_ll, filt.Xs[0], expected_states


def test_extreme_forecast_quarterly_likelihood(extreme_forecast):
    ll, expected_ll, _, _ = extreme_forecast
    assert ll == pytest.approx(expected_ll, rel=0, abs=1e-7)


def test_extreme_forecast_updated_members(extreme_forecast):
    _, _, states, expected_states = extreme_forecast
    np.testing.assert_allclose(states, expected_states, rtol=1e-7, atol=1e-7)


def test_scalar_state_cancellation_with_and_without_fallback():
    x = np.array([[-1e17, 0., 1e17]])
    z = np.array([[1.]])
    # R=1 and var(x)=1e34. For each member the exact conditional update
    # is 1 + (x_i - 1)/(1e34 + 1), which rounds to 1 in Float64.
    def run(digits):
        filt = TEnKF(3, 1, 1, fx=lambda state, eps: (state,), hx=lambda states: states,
                     fallback_decimal_digits=digits)
        filt.multivariate = lambda mean, cov, size, **kwargs: np.zeros((*size, len(mean)))
        updated = filt.batch_filter(z, init_states=x)[:, 0, 0]
        return updated, filt.high_precision_updates

    precision = mp.dps
    updated, count = run(np.int64(80))
    np.testing.assert_array_equal(updated, np.ones(3))
    assert count == 1
    with pytest.raises(ForecastPrecisionRequired, match="Observation 0"):
        run(None)
    assert mp.dps == precision


def test_disabled_retry_keeps_checks_and_only_counts_work_for_the_current_call():
    filt = TEnKF(3, 1, 1, fx=lambda state, eps: (state,), hx=lambda states: states,
                 fallback_decimal_digits=None)
    filt.multivariate = lambda mean, cov, size, **kwargs: np.zeros((*size, len(mean)))
    parts = np.empty(1)
    z = np.array([[1.]])
    with pytest.raises(ForecastPrecisionRequired, match="Observation 0"):
        filt.batch_filter(z, init_states=np.array([[-1e17, 0., 1e17]]), calc_ll=True, ll_contributions=parts)
    assert np.isnan(parts[0])
    assert filt.precision_requests == 1
    assert filt.high_precision_updates == filt.high_precision_completed == filt.high_precision_seconds == 0
    assert np.isfinite(filt.batch_filter(z, init_states=np.array([[1., 2., 3.]]), calc_ll=True))
    assert filt.precision_requests == filt.high_precision_updates == filt.high_precision_seconds == 0


@pytest.mark.parametrize("digits", [0, -5, 50.5, True, "200"])
def test_invalid_fallback_precision(digits):
    with pytest.raises(ValueError, match="positive integer or None"):
        TEnKF(3, 1, 1, fallback_decimal_digits=digits)
