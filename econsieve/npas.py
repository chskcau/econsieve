#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
from tqdm import tqdm
from grgrlib import timeprint
from grgrlib.optimize import cmaes
from .stats import logpdf


def _ensemble_score(points):
    """Whiten differences on the empirical affine support of an ensemble."""
    anomalies = points - points.mean(axis=0)
    scale = np.max(np.abs(anomalies), axis=0)
    scale[scale == 0] = 1
    _, values, directions = np.linalg.svd(anomalies / scale, full_matrices=False)
    cutoff = max(points.shape) * np.finfo(float).eps
    keep = values > cutoff * values[0]
    if keep.all() and len(values) == points.shape[1] and values[-1] > np.finfo(float).eps**0.25 * values[0]:
        return None
    basis = directions[keep]
    std = values[keep] / np.sqrt(len(points) - 1)
    location_scale = np.max(np.abs(points), axis=0) / scale
    null_projector = np.abs(np.eye(len(scale)) - basis.T @ basis)
    abs_basis = np.abs(basis)

    def score(state, target):
        diff = (state - target) / scale
        coords = basis @ diff
        if len(coords) < len(diff):
            residual = diff - coords @ basis
            propagation = null_projector @ (location_scale + np.abs(diff))
            tolerance = cutoff * (propagation + np.abs(diff) + np.abs(coords) @ abs_basis)
            if np.any(np.abs(residual) > tolerance):
                return np.inf
        # CMA-ES compares candidates at a fixed date; the normalizer is constant.
        return 0.5 * np.sum((coords / std)**2)

    return score


def npas(self, func=None, X=None, init_states=None, vals=None, covs=None, nsamples=False, bound_sigma=4, frtol=1e-5, seed=0, verbose=True, **cmaes_args):
    """Nonlinear Path-Adjustment Smoother. 

    Assumes that either `X` (a time series of ensembles) is given (or can be taken from the filter `self` object), or that the time series vals and covs are given. From the filter object, also `Q` (the covariance matrix of shocks) and the transition function `t_func(state, shock_innovations)` must be provided.
    ...

    Parameters
    ----------
    X : array, optional
        a time series of ensembles. Either this, or vals and covs have to be provided
    vals : array, optional
        the series of ensemble values, probably you want the means. Either this together with the covs, or X has to be provided
    covs : array, optional
        the series of ensemble covariances. Either this together with `vals`, or `X` has to be provided
    bound_sigma : int, optional
        the number of standard deviations included in the box constraint of the global optimizer

    Returns
    -------
    X : array 
        the smoothed vals
    covs : array 
        the covariances
    res : array 
        the smoothed/estimated exogenous innovations
    flag : bool
        an error flag of the transition function
    """

    if verbose:
        st = time.time()

    np.random.seed(seed)

    if func is None:
        func = self.t_func

    # X must be time series of ensembles of x dimensions
    if X is None:
        X = np.rollaxis(self.Ss, 2)

    scores = [None] * X.shape[1]
    if covs is None:
        covs = np.empty((X.shape[1], X.shape[2], X.shape[2]))
        for i in range(X.shape[1]):
            covs[i, :, :] = np.cov(X[:, i, :].T)
            scores[i] = _ensemble_score(X[:, i, :])

    bound = np.sqrt(np.diag(self.Q))*bound_sigma

    def target(eps, x, mean, cov, score):

        state, flag = func(x, 2*bound*eps)
        if flag:
            return np.inf

        return score(state, mean) if score is not None else -logpdf(state, mean=mean, cov=cov)

    wrap = tqdm if verbose else lambda x: x
    owrap = wrap if nsamples > 1 else lambda x: x
    iwrap = wrap if nsamples else lambda x: x

    # the smoothest way to do this would be rejection sampling. But  max(p(x) / q(x)) is unknown and expensive to evaluate

    if not nsamples:
        X[0] = np.mean(X, axis=0)
        nsamples = 1

    # preallocate
    res = np.empty((nsamples, len(self.Z)-1, self.dim_z))
    flag = False

    for n, s in enumerate(owrap(X[:nsamples])):

        if init_states is None:
            x = init = X[n][0]
        else:
            x = init = init_states[n]

        for t in iwrap(range(s.shape[0] - 1)):

            def func_cmaes(eps): return target(eps, x, s[t+1], covs[t+1], scores[t+1])

            eps0 = np.zeros(self.dim_z)

            res_cma = cmaes(func_cmaes, eps0, 0.1,
                            verbose=verbose > 1, frtol=frtol, **cmaes_args)
            eps = res_cma[0]*bound*2
            res[n][t] = eps

            x, fflag = func(x, eps)
            flag |= fflag

        if flag and verbose:
            print('[npas:]'.ljust(15, ' ') +
                  'Transition function returned error.')

        if not nsamples and verbose:
            print('[npas:]'.ljust(15, ' ')+'Extraction took ',
                  timeprint(time.time() - st, 3))

    return init, res, flag
