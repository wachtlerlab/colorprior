#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 20.04.21 21:37
"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.utils import resample as bootstrap


def fit_bstrp(x, y, func, n_bs, p_ci, p0=None):
    """

    :param x:       x values
    :param y:       y values
    :param func:    function for curve fitting
    :param n_bs:    number of bootstrap
    :param p_ci:    probability interval
    :param p0:      initial guess on parameters
    :return:
    """
    bs_ypred = np.zeros((n_bs, y.size))

    # Bootstrap
    for i in range(n_bs):
        xi, yi = bootstrap(x, y)
        par_i, cov_i = curve_fit(func, xi, yi, p0=p0, maxfev=2000)
        bs_ypred[i] = func(x, *par_i)

    # CI from bootstrap results
    lower, upper = np.quantile(bs_ypred, [(1 - p_ci) / 2, 1 - (1 - p_ci) / 2], axis=0)

    params, cov = curve_fit(func, x, y, p0=p0, maxfev=2000)

    # Use least square (believe the noise is Gaussian)
    #     noise = np.std(y - ypred)
    #     predictions = np.array([np.random.normal(ypred, noise) for j in range(n_bs)])
    #     lower,upper = np.quantile(predictions, [1 - p_ci, p_ci], axis = 0)

    fit = {'params': params,
           'bounds': [lower, upper]}

    return fit
