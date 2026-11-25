"""High-precision conditioning of a single empirical Gaussian forecast."""

import numpy as np
from mpmath import mp


def _whitener(ctx, covariance, innovation, observation, mean, members, positive_noise):
    """Return a whitener and log volume, or None for an off-support observation."""
    dim = covariance.rows
    scales = [ctx.sqrt(covariance[i, i]) or ctx.one for i in range(dim)]
    unscale = ctx.diag([1 / scale for scale in scales])
    scaled = unscale * covariance * unscale
    try:
        lower = ctx.cholesky(scaled)
    except ValueError:
        if positive_noise:
            raise  # A positive definite R guarantees full support.
    else:
        return lower**-1 * unscale, sum(ctx.log(lower[i, i] * scales[i]) for i in range(dim))

    # A semidefinite R may leave exact linear restrictions on observations.
    values, vectors = ctx.eigsy(scaled)
    tol = dim * ctx.eps * max(1, values[dim - 1])
    if values[0] < -tol:
        raise ValueError("Forecast covariance has a negative eigenvalue.")
    keep = [i for i in range(dim) if values[i] > tol]
    basis = vectors[:, dim - len(keep):]
    v = unscale * innovation
    # Allow Float64 input roundoff when checking membership in the support.
    error = v - basis * (basis.T * v)
    abs_basis = basis.apply(abs)
    magnitude = unscale * (mean.apply(abs) + observation.apply(abs)) + v.apply(abs)
    magnitude += abs_basis * (abs_basis.T * v.apply(abs))
    tolerance = (members + dim) * float(np.finfo(float).eps) * magnitude
    if any(abs(error[i]) > tolerance[i] for i in range(dim)):
        return None
    if not keep:
        return ctx.matrix(0, dim), ctx.zero
    whitener = ctx.diag([1 / ctx.sqrt(values[i]) for i in keep]) * basis.T * unscale
    # Convert the support volume back to the original observation units.
    # Normalize the columns so that tiny physical units alone cannot cause
    # the Cholesky rank check to discard a valid direction.
    support = ctx.diag(scales) * basis
    lengths = [ctx.norm(support[:, i]) for i in range(len(keep))]
    normalized = support * ctx.diag([1 / length for length in lengths])
    lower = ctx.cholesky(normalized.T * normalized)
    log_volume = sum(ctx.log(values[j]) / 2 + ctx.log(lengths[i] * lower[i, i])
                     for i, j in enumerate(keep))
    return whitener, log_volume


def precise_update(X, Y, z, noise, R, positive_noise, calc_ll, digits):
    """Recompute from the uncentered binary inputs, rounding only the final results."""
    ctx = mp.clone()
    ctx.dps = digits
    x, y, observation, perturbations, r = (
        ctx.matrix(np.asarray(a, dtype=float).tolist()) for a in (X, Y, z, noise, R)
    )
    n = y.cols
    ones = ctx.ones(n, 1)
    mean = y * ones / n
    yc = y - mean * ones.T
    xc = x - (x * ones / n) * ones.T
    covariance = yc * yc.T / (n - 1) + r
    innovation = observation - mean
    decomposition = _whitener(ctx, covariance, innovation, observation, mean, n, positive_noise)
    if decomposition is None:
        return None, -np.inf
    whiten, log_volume = decomposition
    cross = xc * yc.T / (n - 1)
    residuals = observation * ones.T - y - perturbations
    updated = x + (cross * whiten.T) * (whiten * residuals)
    ll = 0.
    if calc_ll:
        dev = whiten * innovation
        ll = float(-(whiten.rows * ctx.log(2 * ctx.pi) + ctx.fdot(dev, dev)) / 2 - log_volume)
    return np.array(updated.tolist(), dtype=float), ll
