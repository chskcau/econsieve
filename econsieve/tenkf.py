#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from functools import partial
from inspect import Parameter, signature
from scipy.linalg import sqrtm
from scipy.stats.qmc import MultivariateNormalQMC, LatinHypercube
from grgrlib.linalg import tinv, nearest_psd
from .stats import logpdf


def multivariate_dispatch(engine):

    # Override a pre-bound seed (or rng) without changing other factory options.
    params = signature(engine).parameters
    if 'rng' in params and not (isinstance(engine, partial) and 'seed' in engine.keywords):
        seed_kw = 'rng'
    elif 'seed' in params or any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
        seed_kw = 'seed'
    else:
        seed_kw = None

    def multivariate(mean, cov, size, stream=None):
        ndim = len(mean)
        size = (size,) if isinstance(size, int) else size
        nsample = np.prod(size)

        # define target distribution
        scov = np.real(sqrtm(cov))
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

    name = 'TEnKF'

    def __init__(self, N, dim_x=None, dim_z=None, fx=None, hx=None, qmc_engine=None, seed=None):

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.t_func = fx
        self.o_func = hx

        self.N = N
        self.seed = seed

        self.R = np.eye(self.dim_z)
        self.Q = np.eye(self.dim_x)
        self.P = np.eye(self.dim_x)
        self.x = np.zeros(self.dim_x)

        # enforce deterministic sampling by default
        seed_default_qmc_engine = 0 if seed is None else seed
        qmc_engine = partial(LatinHypercube, seed=seed_default_qmc_engine) if qmc_engine is None else qmc_engine
        self.multivariate = multivariate_dispatch(qmc_engine)


    def batch_filter(self, Z, init_states=None, seed=None, store=False, calc_ll=False, verbose=False):
        """Batch filter.

        Runs the TEnKF on the full dataset. The call seed overrides the filter
        seed (default 0). A custom QMC factory may accept a seed or rng keyword.
        A dimension-only factory must return reproducible points for repeatable
        results; the filter independently shuffles them for each stream.
        """

        # store time series for later
        self.Z = Z

        dim_x = self.dim_x
        dim_z = self.dim_z
        N = self.N

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
        if init_states is None:
            X = self.multivariate(self.x, self.P, N, stream=initial_rng).T
        else:
            X = np.array(init_states, dtype=float, copy=True)

        self.Xs = np.empty((Z.shape[0], dim_x, N))

        for nz, z in enumerate(Z):

            # predict
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
            S = np.cov(Y) + self.R
            if calc_ll:
                # Evaluate the forecast before updating X: Y can be a view of X.
                y = z - np.mean(Y, axis=1)
                ll += logpdf(x=y, mean=np.zeros(dim_z), cov=S)
            X += X_bar @ Y_bar.T @ np.linalg.inv((N-1)
                                                 * S) @ (ZZ - Y - mus[nz].T)

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
