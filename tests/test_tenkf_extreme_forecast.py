"""One archived forecast update, independent of its originating model and solver."""

from pathlib import Path

import numpy as np
import pytest
from mpmath import mp

from econsieve import ForecastPrecisionRequired, TEnKF


def test_precise_forecast_retains_information_lost_at_float64_boundary():
    # Three members y=(a*x, a*x+1), x=(-1,0,1). R=I. The orthogonal
    # forecast difference is exactly one even when a binary64 ULP exceeds one.
    ctx = mp.clone()
    ctx.dps = 200
    a = 2**60
    initial = np.array([[-1., 0., 1.]])
    noise = np.array([[0.25, -.5], [.75, .125], [-.25, .5]])
    seen = []

    def precise(states, shocks):
        seen.append((states.copy(), shocks.copy()))
        q = np.array([[ctx.mpf(v) for v in states[0]]], dtype=object)
        return q, np.concatenate((a*q, a*q+1))

    def fast(state, shock):
        y = float(a) * state[0]
        return (state, np.array([y, y + 1.])), 0

    f = TEnKF(3, 1, 2, fx=fast)
    f.Q = np.eye(1)
    f.multivariate = lambda mean, cov, size, **kw: (noise if cov is f.R else initial.T)[None]
    z = np.array([[1., 3.]])
    terms = np.empty(1)
    ll = f.batch_filter(z, init_states=initial, calc_ll=True, store=True,
                        ll_contributions=terms, precise_forecast=precise)
    # Cov(y) + I has eigenvalues 1 and 2*a**2+1 with mean (0,1).
    v = 2*a**2 + 1
    expected = -np.log(2*np.pi) - (np.log(float(v)) + .5 + 4.5/v)/2
    assert ll == pytest.approx(expected, abs=1e-12)
    assert terms[0] == ll
    np.testing.assert_allclose(f.Xs[0], (initial + a * (3 - noise.sum(axis=1))) / float(v),
                               rtol=1e-14, atol=0)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], initial)
    np.testing.assert_array_equal(seen[0][1], initial.T)


def test_noiseless_scalar_forecast_with_large_intercept_and_small_spread():
    ctx = mp.clone()
    ctx.dps = 200
    level = 2**57
    offsets = np.array([-31., 1., 31.])

    def precise(states, shocks):
        return states, np.array([[ctx.mpf(level) + float(x) for x in shocks[:, 0]]], dtype=object)

    f = TEnKF(3, 1, 1, fx=lambda state, eps: ((state, level+eps), 0), precise_forecast=precise)
    f.R[:] = 0.
    f.multivariate = lambda m, cov, s, **kw: (offsets if cov is f.Q else np.zeros(3)).reshape(1, 3, 1)
    actual = f.batch_filter(np.array([[float(level)]]), init_states=np.ones((1, 3)), calc_ll=True)
    variance = offsets.var(ddof=1)
    expected = -.5 * (np.log(2*np.pi*variance) + offsets.mean()**2/variance)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert f.high_precision_completed == 1


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
