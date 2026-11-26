#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from functools import partial
from inspect import Parameter, signature
from time import perf_counter
from scipy.linalg import sqrtm, solve_triangular
from scipy.stats.qmc import MultivariateNormalQMC, LatinHypercube
from grgrlib.linalg import tinv
from .stats import logpdf
from ._tenkf_precision import precise_update


_SQRT_EPS = np.sqrt(np.finfo(float).eps)
_RETRY_IN_HIGH_PRECISION = object()


class ForecastNumericalError(ArithmeticError):
    """The ensemble forecast could not be evaluated reliably in finite precision."""


class ForecastPrecisionRequired(ForecastNumericalError):
    """This observation needs a high-precision update, which the caller disabled."""


def _measurement_root(R, dim):
    if R.shape != (dim, dim) or not np.isfinite(R).all() or not np.array_equal(R, R.T):
        raise ValueError("Measurement covariance R must be finite, symmetric and match dim_z.")
    try:
        return np.linalg.cholesky(R), True
    except np.linalg.LinAlgError:
        diagonal = R.diagonal()
        if np.any(diagonal < 0) or np.any(R[diagonal == 0] != 0):
            raise ValueError("Measurement covariance R must be positive semidefinite.")
        scale = np.sqrt(diagonal)
        scale[scale == 0] = 1
        values, vectors = np.linalg.eigh((R / scale[:, None]) / scale)
        tol = dim * np.finfo(float).eps * np.max(np.abs(values))
        if np.any(values < -tol):
            raise ValueError("Measurement covariance R must be positive semidefinite.")
        return scale[:, None] * vectors[:, values > tol] * np.sqrt(values[values > tol]), False


def _factor_update(X, Y, z, noise, root, positive_noise, calc_ll):
    """Condition using a factor of cov(Y) + R before rounding can erase R."""
    n = X.shape[1]
    mean = Y.mean(axis=1)
    yc = (Y - mean[:, None]) / np.sqrt(n - 1)
    xc = (X - X.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)
    factor = np.column_stack((yc, root))
    # Equilibrate rows so a small variance in different units retains its rank.
    scale = np.max(np.abs(factor), axis=1)
    scale[scale == 0] = 1
    factor /= scale[:, None]
    innovation = (z - mean) / scale
    residuals = ((z[:, None] - Y) - noise) / scale[:, None]
    if positive_noise:
        q, upper = np.linalg.qr(factor.T, mode="reduced")
        if np.linalg.cond(upper) > 1 / _SQRT_EPS:
            return _RETRY_IN_HIGH_PRECISION, None
        whiten = partial(solve_triangular, upper.T, lower=True)
        update = (xc @ q[:n]) @ whiten(residuals)
        if not calc_ll:
            return update, 0.
        dev = whiten(innovation)
        log_volume = np.log(scale).sum() + np.log(np.abs(upper.diagonal())).sum()
        rank = len(z)
    else:
        u, s, vt = np.linalg.svd(factor, full_matrices=False)
        if s[-1] < _SQRT_EPS * s[0]:
            return _RETRY_IN_HIGH_PRECISION, None
        eps = np.finfo(float).eps
        keep = s > max(factor.shape) * eps * s[0]
        basis = u[:, keep]
        outside = innovation - basis @ (basis.T @ innovation)
        tol = max(factor.shape) * eps * (np.abs(innovation) + np.abs(basis) @ np.abs(basis.T @ innovation)
                                         + (np.abs(mean) + np.abs(z)) / scale)
        if np.any(np.abs(outside) > tol):
            return None, -np.inf
        update = (xc @ vt[keep, :n].T) @ ((basis.T @ residuals) / s[keep, None])
        if not calc_ll:
            return update, 0.
        dev = (basis.T @ innovation) / s[keep]
        # Volume on the affine support in the original observation units.
        volume_factor = np.linalg.qr(scale[:, None] * basis, mode='r')
        log_volume = np.log(s[keep]).sum() + np.log(np.abs(volume_factor.diagonal())).sum()
        rank = int(keep.sum())
    return update, -0.5 * (rank * np.log(2*np.pi) + dev @ dev) - log_volume


def multivariate_dispatch(engine):

    # Override a pre-bound seed (or rng) without changing other factory options.
    params = signature(engine).parameters
    if 'rng' in params and not (isinstance(engine, partial) and 'seed' in engine.keywords):
        seed_kw = 'rng'
    elif 'seed' in params or any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
        seed_kw = 'seed'
    else:
        seed_kw = None

    def multivariate(mean, cov, size, stream=None, cov_root=None):
        ndim = len(mean)
        size = (size,) if isinstance(size, int) else size
        nsample = np.prod(size)

        # define target distribution
        scov = cov_root
        if scov is None:
            scov = np.real(sqrtm(cov)) if np.any(cov) else np.zeros_like(cov)
        options = {}
        if stream is not None and seed_kw is not None:
            options[seed_kw] = int(stream.integers(2**32))
        dist = MultivariateNormalQMC(mean, cov_root=scov, engine=engine(ndim, **options))

        # draw sample
        res = dist.random(nsample)
        if stream is not None:
            # Pair points independently, even for engines with fixed point order.
            stream.shuffle(res, axis=0)
        return np.reshape(res, (*size, ndim))

    return multivariate


class TEnKF(object):

    """Transposed ensemble Kalman filter.

    fallback_decimal_digits sets the number of significant decimal digits used
    to recompute a numerically suspect observation update (default 200).
    None keeps the checks and raises ForecastPrecisionRequired if a retry is needed.
    precise_forecast, when supplied, takes the previous member states (dim_x, N)
    and exactly the same process draws (N, dim_eps). It returns the unrounded
    state and observation forecasts in (dimension, N) order and must be deterministic.
    """

    name = 'TEnKF'
    precise_forecast = None  # Also the default for previously serialized filters.

    def __init__(self, N, dim_x=None, dim_z=None, fx=None, hx=None, qmc_engine=None, seed=None,
                 *, fallback_decimal_digits=200, precise_forecast=None):

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.t_func = fx
        self.o_func = hx
        self.precise_forecast = precise_forecast

        self.N = N
        self.seed = seed
        if fallback_decimal_digits is not None:
            if (isinstance(fallback_decimal_digits, bool)
                    or not isinstance(fallback_decimal_digits, (int, np.integer))
                    or fallback_decimal_digits < 1):
                raise ValueError("fallback_decimal_digits must be a positive integer or None.")
            fallback_decimal_digits = int(fallback_decimal_digits)
        self.fallback_decimal_digits = fallback_decimal_digits
        self.precision_requests = 0
        self.high_precision_updates = 0
        self.high_precision_completed = 0
        self.high_precision_seconds = 0.0

        self.R = np.eye(self.dim_z)
        self.Q = np.eye(self.dim_x)
        self.P = np.eye(self.dim_x)
        self.x = np.zeros(self.dim_x)

        # enforce deterministic sampling by default
        seed_default_qmc_engine = 0 if seed is None else seed
        qmc_engine = partial(LatinHypercube, seed=seed_default_qmc_engine) if qmc_engine is None else qmc_engine
        self.multivariate = multivariate_dispatch(qmc_engine)


    def batch_filter(self, Z, init_states=None, seed=None, store=False, calc_ll=False, verbose=False,
                     *, ll_contributions=None, precise_forecast=None, initial_cov_root=None):
        """Batch filter.

        Runs the TEnKF on the full dataset. The call seed overrides the filter
        seed (default 0). A custom QMC factory may accept a seed or rng keyword.
        A call-specific precise_forecast overrides the constructor callback.
        initial_cov_root may supply a square right factor S of the stationary
        covariance (S.T @ S = P) for the initial QMC draws. This preserves
        information available to a model that formed P from smaller factors.
        A dimension-only factory must return reproducible points for repeatable
        results; the filter independently shuffles them for each stream.
        """

        self.precision_requests = 0
        self.high_precision_updates = 0
        self.high_precision_completed = 0
        self.high_precision_seconds = 0.0

        # Optional caller-owned float64 buffer. Preserve the evaluated prefix on rejection.
        if ll_contributions is not None:
            if (not calc_ll or not isinstance(ll_contributions, np.ndarray)
                    or ll_contributions.dtype != np.float64 or ll_contributions.shape != (len(Z),)):
                raise ValueError("ll_contributions requires calc_ll and one float64 entry per observation.")
            ll_contributions.fill(np.nan)

        # store time series for later
        self.Z = Z

        dim_x = self.dim_x
        dim_z = self.dim_z
        N = self.N
        if N < 2:
            raise ValueError("The ensemble must have at least two members.")
        try:
            noise_root, positive_noise = _measurement_root(np.asarray(self.R), dim_z)
        except np.linalg.LinAlgError as exc:
            raise ForecastNumericalError("Could not factor measurement covariance R.") from exc

        I1 = np.ones(N)
        I2 = np.eye(N) - np.outer(I1, I1)/N

        # pre allocate
        if store:
            self.Xs = np.empty((Z.shape[0], dim_x, N))
            self.X_priors = np.empty_like(self.Xs)
            self.X_bars = np.empty_like(self.Xs)
            self.X_bar_priors = np.empty_like(self.Xs)

        ll = 0

        seed = self.seed if seed is None else seed
        children = np.random.SeedSequence(0 if seed is None else seed).spawn(3)
        process_rng, measurement_rng, initial_rng = map(np.random.default_rng, children)

        means = np.empty((Z.shape[0], dim_x))
        covs = np.empty((Z.shape[0], dim_x, dim_x))
        Y = np.empty((dim_z, N))

        size = (len(Z), N)
        epss = self.multivariate(np.zeros(self.Q.shape[0]), self.Q, size, stream=process_rng)
        mus = self.multivariate(np.zeros(dim_z), self.R, size, stream=measurement_rng)
        if initial_cov_root is not None:
            if init_states is not None:
                raise ValueError("Specify either init_states or initial_cov_root.")
            root = np.asarray(initial_cov_root, dtype=float)
            if root.shape != (dim_x, dim_x) or not np.isfinite(root).all():
                raise ValueError("initial_cov_root must be a finite square factor of P.")
            X = self.multivariate(self.x, self.P, N, stream=initial_rng, cov_root=root).T
        elif init_states is None:
            X = self.multivariate(self.x, self.P, N, stream=initial_rng).T
        else:
            X = np.array(init_states, dtype=float, copy=True)

        self.Xs = np.empty((Z.shape[0], dim_x, N))
        forecast_again = self.precise_forecast if precise_forecast is None else precise_forecast
        noise_scale = np.sqrt(self.R.diagonal())

        for nz, z in enumerate(Z):

            # predict
            prior_members = X.copy() if forecast_again is not None else None
            for i in range(N):
                eps = epss[nz, i]
                if self.o_func is None:
                    X[:, i], Y[:, i] = self.t_func(X[:, i], eps)[0]
                else:
                    X[:, i] = self.t_func(X[:, i], eps)[0]

            if self.o_func is not None:
                Y = self.o_func(X.T).T

            if store:
                self.X_priors[nz, :, :] = X

            # update
            X_bar = X @ I2
            Y_bar = Y @ I2
            ZZ = np.outer(z, I1)
            try:
                if not np.isfinite(Y).all() or not np.isfinite(X).all():
                    raise ForecastNumericalError(f"Non-finite ensemble forecast at observation {nz}.")
                with np.errstate(over='ignore', invalid='ignore'):
                    S = np.cov(Y) + self.R
                    scaled_cov = (N-1) * S
                try:
                    inverse = np.linalg.inv(scaled_cov) if np.isfinite(scaled_cov).all() else None
                except np.linalg.LinAlgError:
                    inverse = None
                # The inverse is applied to a Gram matrix. Switch to its QR
                # factor before moderate conditioning contaminates later states.
                condition = np.inf if inverse is None else np.sum(np.abs(scaled_cov * inverse.T))
                contribution = 0.
                if not np.isfinite(condition) or condition > 1 / np.sqrt(_SQRT_EPS):
                    update, contribution = _factor_update(
                        X, Y, z, mus[nz].T, noise_root, positive_noise, calc_ll)
                else:
                    if calc_ll:
                        y = z - np.mean(Y, axis=1)
                        contribution = logpdf(x=y, mean=np.zeros(dim_z), cov=S)
                    update = X_bar @ Y_bar.T @ inverse @ (ZZ - Y - mus[nz].T)
                updated = None
                # Compare the rounding scale of the uncentered observation
                # forecasts to the measurement noise in the same physical units.
                resolution = noise_scale
                if not positive_noise:
                    # For a noiseless observation, compare with the predicted
                    # spread. A shared large intercept can erase that spread.
                    resolution = np.where(noise_scale > 0, noise_scale, np.sqrt(np.maximum(S.diagonal(), 0)))
                large_forecast = np.any(_SQRT_EPS * np.max(abs(Y), axis=1) > resolution)
                retry = update is _RETRY_IN_HIGH_PRECISION
                if update is not None and not retry:
                    updated = X + update
                    # Losing half the significand in X + update can erase a
                    # member's posterior state even in a scalar observation.
                    retry = (not np.isfinite(updated).all()
                             or np.any(np.abs(updated) < _SQRT_EPS * np.abs(X)))
                retry |= large_forecast
                if retry:
                    self.precision_requests += 1
                    if self.fallback_decimal_digits is None:
                        raise ForecastPrecisionRequired(
                            f"Observation {nz} requires a high-precision forecast update "
                            "with fallback_decimal_digits=None."
                        )
                    self.high_precision_updates += 1
                    started = perf_counter()
                    try:
                        px, py = X, Y
                        if forecast_again is not None:
                            px, py = forecast_again(prior_members, epss[nz])
                            if np.shape(px) != X.shape or np.shape(py) != Y.shape:
                                raise ValueError("Incorrect dimensions returned by precise_forecast.")
                        updated, contribution = precise_update(
                            px, py, z, mus[nz].T, self.R, positive_noise, calc_ll, self.fallback_decimal_digits)
                    except (ValueError, ZeroDivisionError) as exc:
                        raise ForecastNumericalError(
                            f"Ensemble update at observation {nz} failed at "
                            f"{self.fallback_decimal_digits} decimal digits: {exc}") from exc
                    else:
                        self.high_precision_completed += 1
                    finally:
                        self.high_precision_seconds += perf_counter() - started
                if ll_contributions is not None:
                    ll_contributions[nz] = contribution
                if updated is None:
                    if calc_ll and not store:
                        self.ll = -np.inf
                        return self.ll
                    raise ValueError(f"Observation {nz} is outside the forecast support.")
                X[:] = updated
                ll += contribution
                if not np.isfinite(X).all() or np.isnan(ll) or np.isposinf(ll):
                    raise ForecastNumericalError(f"Non-finite ensemble update at observation {nz}.")
            except (np.linalg.LinAlgError, FloatingPointError, RuntimeWarning) as exc:
                raise ForecastNumericalError(f"Numerical failure in the ensemble update at observation {nz}: {exc}") from exc

            if store:
                self.X_bar_priors[nz, :, :] = X_bar
                self.X_bars[nz, :, :] = X @ I2
            if store or not calc_ll:
                self.Xs[nz, :, :] = X

        if calc_ll:
            self.ll = ll
            return ll
        else:
            return np.rollaxis(self.Xs, 2)

    def rts_smoother(self, means=None, covs=None, rcond=1e-14):

        S = self.Xs[-1]
        Ss = self.Xs.copy()

        for i in reversed(range(self.Xs.shape[0] - 1)):

            J = self.X_bars[i] @ tinv(self.X_bar_priors[i+1])
            S = self.Xs[i] + J @ (S - self.X_priors[i+1])

            Ss[i, :, :] = S

        self.Ss = Ss

        return np.rollaxis(Ss, 2)
