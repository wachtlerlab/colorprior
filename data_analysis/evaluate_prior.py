#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 12.08.21 17:22

Evaluate a prior model by simulating the bias and measuring the corresponding Likelihood or sum of squared errors.

"""
import numpy as np
from data_analysis.fit_curves import FitCumNormal
from data_analysis.get_MAP import get_MAP
import pandas as pd

x_grid = np.load("data_analysis/model_estimates/x_grid.npy")


def simul_estimate(lh_data, prior, lh_lik):
    """

    :param lh_data:
    :param prior:
    :param lh_lik:
    :return:
    """
    unique_pairs = lh_data[['standard_stim', 'test_stim', 'actual_intensity']]. \
        drop_duplicates().sort_values(by=['standard_stim', 'test_stim'])
    stim_l = unique_pairs['standard_stim'].values.astype('float')
    stim_h = unique_pairs['test_stim'].values.astype('float')

    likelihoods_l, likelihoods_h = lh_lik[0], lh_lik[1]

    # Simulate for each stimulus value
    _, MAP_x_l = get_MAP(prior, likelihoods_l, x_grid)
    _, MAP_x_h = get_MAP(prior, likelihoods_h, x_grid)

    simul_n = np.shape(MAP_x_l)[1]
    simul_res = unique_pairs.copy()
    simul_res['intens'] = stim_h - stim_l
    simul_res['resp'] = lh_data.groupby(['standard_stim', 'test_stim']) \
        ['resp_as_larger'].apply(lambda g: g.sum() / len(g)).values

    simul_res['simul_resp'] = np.sum((MAP_x_h > MAP_x_l), axis=1) / simul_n
    simul_fit = simul_res.groupby('standard_stim').apply(
        lambda x: FitCumNormal(x['intens'], x['simul_resp'], guess=[0, 5], expectedMin=0.0, lapse=0.0))
    return simul_res, simul_fit


# def each_loglik(lh_data, simul_res, simul_fit):
def each_loglik(lh_data, simul_fit):

    log_lik_list = []

    # for k, grp in simul_res.groupby('standard_stim'):
    for k in list(simul_fit.keys()):
        stim_data = lh_data.query("standard_stim == @k")
        true_resp = stim_data['resp_as_larger']
        pred_resp = simul_fit.loc[k].eval(stim_data['actual_intensity'])

        log_lik = np.sum(true_resp * np.log(pred_resp) + (1 - true_resp) * np.log(1 - pred_resp))
        log_lik_list.append(log_lik)
    return log_lik_list


def each_sqrt_error(simul_res, simul_fit):
    sqrt_error_list = []

    for k, grp in simul_res.groupby('standard_stim'):
        true_resp = grp['resp']
        pred_resp = simul_fit.loc[k].eval(grp['intens'])
        sqrt_error = sum((pred_resp - true_resp) ** 2)
        sqrt_error_list.append(sqrt_error)

    return sqrt_error_list


"""
# Example: evaluate a uniform prior

def prior_uniform():
    y = 1./(len(x_grid))
    return np.repeat(y, len(x_grid))
    
sub = 's5'
sub_in_data = 's05'
all_lh_data = pd.read_csv('data/all_sel_data.csv').query('condition == "LH"')
all_lh_estimates = pd.read_csv('data_analysis/pf_estimates/all_estimates.csv').query('condition == "LH"')
test_data = all_lh_data.query('subject == @sub_in_data')

flat_prior = prior_uniform()
sub_path = "data_analysis/model_estimates/s5/s5"
lik_l = np.load(sub_path + "_likelihoods_l.npy")
lik_h = np.load(sub_path + "_likelihoods_h.npy")
simul_res_flat, simul_fit_flat = simul_estimate(test_data, flat_prior, [lik_l, lik_h])
loglik = np.sum(each_loglik(simul_res_flat, simul_fit_flat)
"""