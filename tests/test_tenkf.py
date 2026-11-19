from functools import partial

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube

from econsieve import TEnKF


def observe(states):
    return states


def test_one_period_posterior():
    filt = TEnKF(1024, 1, 1, fx=lambda x, eps: (eps,), hx=observe, seed=42)
    ensemble = filt.batch_filter(np.array([[2.0]]))[:, 0, 0]

    # With Q = R = 1 the Kalman gain is 1/2.
    assert ensemble.mean() == pytest.approx(1.0, abs=0.05)
    assert ensemble.var(ddof=1) == pytest.approx(0.5, abs=0.06)


def test_two_period_log_likelihood():
    filt = TEnKF(1024, 1, 1, fx=lambda x, eps: (0.8 * x + eps,), hx=observe, seed=42)
    filt.P[:] = 0  # Start at x_0 = 0, so x_1 = eps_1.
    observations = np.array([[2.0], [-3.2]])
    ll = filt.batch_filter(observations, calc_ll=True)

    # y_2 | y_1 has mean 0.8 and variance 0.8**2 * 0.5 + Q + R.
    exact = norm.logpdf(2.0, scale=np.sqrt(2)) + norm.logpdf(-3.2, 0.8, np.sqrt(2.32))
    assert ll == pytest.approx(exact, abs=0.20)


def test_log_likelihood_with_small_positive_measurement_variance():
    # Identical observation forecasts have zero sample covariance, so S = R exactly.
    def zero_observations(states):
        return np.zeros((len(states), 3))

    filt = TEnKF(8, 1, 3, fx=lambda x, eps: (eps,), hx=zero_observations, seed=42)
    filt.R = np.diag([1.0, 1.0, 1e-12])

    ll_a = filt.batch_filter(np.array([[0.0, 0.0, 0.0]]), calc_ll=True)
    ll_b = filt.batch_filter(np.array([[0.0, 0.0, 0.01]]), calc_ll=True)

    expected_a = -0.5 * (3 * np.log(2 * np.pi) + np.log(1e-12))
    np.testing.assert_allclose(ll_a, expected_a, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ll_b - ll_a, -50_000_000, rtol=0, atol=1e-7)


@pytest.mark.parametrize("engine", [
    None,
    partial(LatinHypercube, scramble=False, seed=7),
    lambda d: LatinHypercube(d, seed=7),
])
def test_streams_replay_and_do_not_depend_on_init_states(engine):
    # Ignore x so that any change in the output must come from the shock streams.
    filt = TEnKF(128, 1, 1, fx=lambda x, eps: (eps,), hx=observe, qmc_engine=engine, seed=42)
    observations = np.array([[2.0], [-3.2]])
    first = filt.batch_filter(observations).copy()
    assert first[:, 0, 0].var(ddof=1) > 0.2

    other = filt.batch_filter(observations, seed=43)
    assert not np.array_equal(first, other)
    np.testing.assert_array_equal(filt.batch_filter(observations, seed=42), first)

    initial = np.zeros((1, filt.N))
    np.testing.assert_array_equal(filt.batch_filter(observations, init_states=initial), first)
    np.testing.assert_array_equal(initial, np.zeros_like(initial))


def test_replay_with_sampled_initial_states_and_custom_factory():
    seeds = []

    def engine(d, *, rng):
        seeds.append(rng)
        return LatinHypercube(d, seed=rng)

    filt = TEnKF(64, 1, 1, fx=lambda x, eps: (x + eps,), hx=observe, qmc_engine=engine)
    observations = np.array([[2.0], [-3.2]])
    first_ll = filt.batch_filter(observations, calc_ll=True, store=True)
    first_ensemble = filt.Xs.copy()
    assert len(set(seeds)) == 3

    filt.batch_filter(-observations, seed=123)
    assert filt.batch_filter(observations, calc_ll=True, store=True) == first_ll
    np.testing.assert_array_equal(filt.Xs, first_ensemble)
    assert seeds[-3:] == seeds[:3]
