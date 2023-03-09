#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 10.08.21 18:38
"""
import numpy as np


def get_MAP(prior, likelihoods, x_grid):
    """
    Computer posterior and MAP estimates.

    :param prior:           prior values
    :param likelihoods:     likelihood values
    :return:                estimates of MAP
    """
    marginal_posterior = prior * likelihoods
    posterior = np.array([(p.T / np.sum(p, axis=1)).T for p in marginal_posterior])  # normalize each posterior
    MAP = np.array([np.max(p, axis=1) for p in posterior])
    MAP_x = np.array([x_grid[np.argmax(p, axis=1)] for p in posterior])
    return MAP, MAP_x