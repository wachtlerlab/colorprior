#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 02.08.21 10:53

Module for estimating the prior.
Before running, likelihood estimates should be created.


"""

import numpy as np
import pandas as pd
import os
from data_analysis.fit_bstrp import fit_bstrp
import scipy.stats as st
from scipy.special import i0
from data_analysis.fit_curves import FitCumNormal
from data_analysis.get_MAP import get_MAP
from psychopy import data
from scipy.optimize import curve_fit, minimize, least_squares, brute, fmin, differential_evolution
from sklearn.utils import resample as bootstrap
import json
import pickle
from datetime import datetime

# n_sample = 1000


class EstimatePrior:
    """
    Estimate the parameters of a given prior function.

    Strategy:
    1. (should be done before estimating) sample stimulus pairs from measurement distribution
    2. (should be done before estimating) estimate likelihood of the samples
    3. compute MAP of the samples simulation
    4. for each MAP pair, compute intensity and response ratios => one point on a psychometric function
    5. for each stimulus, fit a psychometric function to all simulated data
    6. evaluate the fitting of simulation-based psychometric on true data and optimize the prior parameters accordingly,
       by maximizing the likelihood (sum of all stimuli), or alternatively, minimizing errors (sum of all stimuli)

    """

    def __init__(self, lh_dat, lh_lik, prior_func, par_guess, par_bounds, save_dir=None):
        """
        :param lh_dat:      L-H cross-noise data [dataframe]
        :param lh_lik:      [likelihoods_low, likelihood_high]
        :param prior_func:  prior function
        :param par_guess:   initial guess of the parameters
        :param par_bounds:  bounds of the parameters
        :param save_dir:    data saving path (and partial name), e.g. 'data_analysis/model_estimates/s0/s0'
        """
        self.lh_data = lh_dat
        self.unique_pairs = self.lh_data[['standard_stim', 'test_stim', 'actual_intensity']]. \
            drop_duplicates().sort_values(by=['standard_stim', 'test_stim'])
        self.stim_l = self.unique_pairs['standard_stim'].values.astype('float')
        self.stim_h = self.unique_pairs['test_stim'].values.astype('float')
        self.likelihoods_l, self.likelihoods_h = lh_lik
        self.prior_func = prior_func
        self.par_guess = par_guess
        self.par_bounds = par_bounds
        self.save_dir = save_dir
        # if self.save_dir is not None:
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)

    def sum_neg_loglik(self, pars):
        """
        Compute the sum of negative log likelihood of true data given the simulation-based psychometric function.

        :param pars:    parameters of the prior
        :return:        the sum of negative log likelihood
        """

        prior = self.prior_func(pars)

        # Simulate for each stimulus value
        _, MAP_x_l = get_MAP(prior, self.likelihoods_l, x_grid)
        _, MAP_x_h = get_MAP(prior, self.likelihoods_h, x_grid)

        # Intensity-response pairs of simulation
        simul_n = np.shape(MAP_x_l)[1]
        simul_res = self.unique_pairs.copy()
        simul_res['intens'] = self.stim_h - self.stim_l
        simul_res['resp'] = self.lh_data.groupby(['standard_stim', 'test_stim']) \
            ['resp_as_larger'].apply(lambda g: g.sum() / len(g)).values
        simul_res['simul_resp'] = np.sum((MAP_x_h > MAP_x_l), axis=1) / simul_n

        # Fit to a psychometric function
        simul_fit = simul_res.groupby('standard_stim').apply(
            lambda x: FitCumNormal(x['intens'], x['simul_resp'], guess=[0, 5], expectedMin=0.0, lapse=0.0))

        # Compute the log likelihood of model fitting
        log_lik_list = []
        start_time = datetime.now()
        for k, grp in simul_res.groupby('standard_stim'):
            stim_data = self.lh_data.query("standard_stim == @k")
            true_resp = stim_data['resp_as_larger']
            pred_resp = simul_fit.loc[k].eval(stim_data['actual_intensity'])

            log_lik = np.sum(true_resp * np.log(pred_resp) + (1 - true_resp) * np.log(1 - pred_resp))
            log_lik_list.append(log_lik)

        # # Save simulated data
        if self.save_dir is not None:
            with open(self.save_dir + '_simul_res.pickle', 'wb') as pf:
                pickle.dump(simul_res, pf)
            with open(self.save_dir + '_simul_fit_res.pickle', 'wb') as pf:
                pickle.dump(simul_fit, pf)
            np.array(log_lik_list).dump(self.save_dir + '_simul_log_liks.npy')

        return -1 * sum(log_lik_list)

    def estimate_par_mle(self):
        """
        Find the best-fit parameters of the prior by maximum likelihood estimation (MLE).

        :return:            optimization results
        """
        # min_res = minimize(self.sum_neg_loglik,
        #                    x0=self.par_guess,
        #                    bounds=self.par_bounds,
        #                    method='Nelder-Mead',
        #                    options={'disp': True})  #'L-BFGS-B',
        # res_brute = brute(self.sum_neg_loglik,
        #                   ranges=self.par_bounds,
        #                   disp=True,
        #                   finish=fmin)
        res_diff_evl = differential_evolution(self.sum_neg_loglik,
                                              bounds=self.par_bounds,
                                              disp=True, seed=40)  # seed=40
        print(res_diff_evl)
        # params = res_brute.x0
        params = res_diff_evl.x
        results = {'OptimizeResult': res_diff_evl.message,
                   'params': params.tolist(),
                   'prior': self.prior_func(params).tolist()}
        if self.save_dir is not None:
            json.dump(results, open(self.save_dir + "_optm.json", 'w'))
        return results

    """
    def sum_squared_error(self, pars):
        prior = self.prior_func(pars)

        # Simulate for each stimulus value
        _, MAP_x_l = get_MAP(prior, self.likelihoods_l, x_grid)
        _, MAP_x_h = get_MAP(prior, self.likelihoods_h, x_grid)

        simul_n = np.shape(MAP_x_l)[1]

        simul_res = self.unique_pairs.copy()
        simul_res['intens'] = self.stim_h - self.stim_l
        simul_res['resp'] = self.lh_data.groupby(['standard_stim', 'test_stim']) \
            ['resp_as_larger'].apply(lambda g: g.sum() / len(g)).values

        simul_res['simul_resp'] = np.sum((MAP_x_h > MAP_x_l), axis=1) / simul_n
        simul_fit = simul_res.groupby('standard_stim').apply(
            lambda x: FitCumNormal(x['intens'], x['simul_resp'], guess=[0, 5], expectedMin=0.0, lapse=0.0))
        sqrt_error_list = []
        start_time = datetime.now()

        for k, grp in simul_res.groupby('standard_stim'):
            true_resp = grp['resp']
            pred_resp = simul_fit.loc[k].eval(grp['intens'])
            sqrt_error = sum((pred_resp - true_resp) ** 2)
            sqrt_error_list.append(sqrt_error)
        # print('Simulation fitting Done: ' + str(datetime.now() - start_time))
        if self.save_dir is not None:
            with open(self.save_dir + '_simul_res.pickle', 'wb') as pf:
                pickle.dump(simul_res, pf)
            with open(self.save_dir + '_simul_fit_res.pickle', 'wb') as pf:
                pickle.dump(simul_fit, pf)
        return sum(sqrt_error_list)

    def estimate_par_lsq(self, save2json=True):
        lsq_res = least_squares(self.sum_squared_error,
                                x0=self.par_guess,
                                bounds=self.par_bounds)
        print(lsq_res)
        params = lsq_res.x
        results = {'params': params.tolist(),
                   'prior': self.prior_func(params).tolist()}
        if self.save_dir is not None and save2json:
            json.dump(results, open(self.save_dir + "_prior.json", 'w'))
        return results
    """


def prior_sine(pars):
    """
    A sine model for the prior, with period = 360

    :param pars:
    :return: normalized prior
    """
    a, b, c = pars
    y = a * np.sin(np.deg2rad(x_grid) + b) + a + c
    y = y / sum(y)
    return y


def prior_sine_2peak(pars):
    """
    A sine model for the prior, with period = 180

    :param pars:
    :return: normalized prior
    """
    a, b, c = pars
    y = a * np.sin(2 * np.deg2rad(x_grid) + b) + a + c
    y = y / sum(y)
    return y


def prior_circ_gaussian(pars):
    """
    A von Mises model for the prior.

    :param pars:
    :return: normalized prior
    """
    mu, kappa = pars
    x_rad = x_grid / 180. * np.pi
    y = np.exp(kappa * np.cos(x_rad - mu)) / (2 * np.pi * i0(kappa))
    return y / sum(y)


def prior_circ_gaussian_2peak(pars):
    """
    A von Mises model for the prior, with 2 peaks

    :param pars:
    :return: normalized prior
    """
    # w, mu1, mu2, kappa1, kappa2 = pars
    mu1, mu2, kappa1, kappa2 = pars
    x_rad = x_grid / 180. * np.pi
    y1 = np.exp(kappa1 * np.cos(x_rad - mu1)) / (2 * np.pi * i0(kappa1))
    y2 = np.exp(kappa2 * np.cos(x_rad - mu2)) / (2 * np.pi * i0(kappa2))
    # y = w * y1 + (1-w) * y2
    y = y1 + y2
    return y / sum(y)


def prior_circ_gaussian_2peak_sym(pars):
    """
    A von Mises model for the prior, with 2 peaks

    :param pars:
    :return: normalized prior
    """
    # w, mu1, mu2, kappa1, kappa2 = pars
    mu1, mu2, kappa = pars
    x_rad = x_grid / 180. * np.pi
    y1 = np.exp(kappa * np.cos(x_rad - mu1)) / (2 * np.pi * i0(kappa))
    y2 = np.exp(kappa * np.cos(x_rad - mu2)) / (2 * np.pi * i0(kappa))
    # y = w * y1 + (1-w) * y2
    y = y1 + y2
    return y / sum(y)


def run_estimation(sub_list, func_type, model_path, n_btrp=0):
    """
    Estimate prior for each given subject.

    If bootstrapping:
        Estimation results will be saved in data_analysis/model_estimates/s0/s0_functype_prior_btrp100.json
        Simulated data will be saved.
    Else:
        Estimation results will be saved in data_analysis/model_estimates/s0/s0_functype_prior.json
        Simulated data will NOT be saved.

    :param sub_list:    subject list, e.g.['s1', 's2', 's3']
    :param func_type:   'sine' or 'gauss'
    :param n_btrp:      number of bootstrapping
    :return:
    """

    all_lh_data = pd.read_csv('data/all_sel_data.csv').query("condition == 'LH'")

    for sub in sub_list:
        if sub == 'sAVG':
            sub_lh_data = all_lh_data
        else:
            sub_in_data = sub[0] + '0' + sub[1]
            sub_lh_data = all_lh_data.query('subject == @sub_in_data')

        sub_path = f"{model_path}/{sub}/{sub}"

        lik_l = np.load(sub_path + "_likelihoods_l.npy")
        lik_h = np.load(sub_path + "_likelihoods_h.npy")

        # bounds_mle = [(None, None), (-np.pi, np.pi), (0.0001, 1.)]
        # bounds_lsq = [(-5, -np.pi, 0.0001),
        #               (5, np.pi, 1.)]
        # ranges_brute =(slice(1, 10, 0.5), slice(-np.pi, np.pi, np.pi/36), slice(0.01, 0.99, 0.02))

        if func_type == 'sine':
            prior = prior_sine
            bounds = [(1., 10.), (-np.pi, np.pi), (0.0001, 0.9999)]
            guess = np.array([1, -np.pi / 10, 0.01])
        elif func_type == 'sine_2peak':
            prior = prior_sine_2peak
            bounds = [(1., 10.), (-np.pi, np.pi), (1.0001, 1.9999)]
            guess = np.array([1, -np.pi / 10, 0.01])
        elif func_type == 'gauss':
            prior = prior_circ_gaussian
            bounds = [(-np.pi, np.pi), (0, 10)]
            guess = np.array([-np.pi / 10, 5])
        elif func_type == 'gauss_2peak':
            prior = prior_circ_gaussian_2peak
            bounds = [(0, 1), (-np.pi, np.pi),  (-np.pi, np.pi), (0, 10), (0, 10)]
            guess = np.array([1, -np.pi/10, -np.pi/10+np.pi, 5, 5])
        elif func_type == 'gauss_2peak_ew':
            prior = prior_circ_gaussian_2peak
            bounds = [(-np.pi, np.pi),  (-np.pi, np.pi), (0, 15), (0, 15)]
            guess = np.array([-np.pi/10, -np.pi/10+np.pi, 5, 5])
        elif func_type == 'gauss_2peak_ewk':
            prior = prior_circ_gaussian_2peak_sym
            bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (0, 15)]
            guess = np.array([-np.pi / 10, -np.pi / 10, 5])
        else:
            raise ValueError("Function type not defined!")

        if n_btrp == 0:
            EstimatePrior(sub_lh_data,
                          lh_lik=[lik_l, lik_h],
                          prior_func=prior,
                          par_guess=guess,
                          par_bounds=bounds,
                          save_dir=f"{sub_path}_{func_type}_prior").estimate_par_mle()

        else:
            res_btrp = []
            for i_btrp in range(n_btrp):
                sample_data = sub_lh_data.groupby(['standard_stim', 'actual_intensity']).\
                                sample(frac=1., replace=True, random_state=i_btrp)
                results = EstimatePrior(sample_data,
                                        lh_lik=[lik_l, lik_h],
                                        prior_func=prior,
                                        par_guess=guess,
                                        par_bounds=bounds,
                                        save_dir=f"{sub_path}_{func_type}_prior").estimate_par_mle()

                res_btrp.append(results)

            json_file = sub_path + '_' + func_type + "_prior_btrp" + str(n_btrp) + ".json"
            output_file = open(json_file, 'w', encoding='utf-8')
            for r in res_btrp:
                json.dump(r, output_file)
                output_file.write("\n")


"""
model_path = "data_analysis/model_estimates_v2_test"
x_grid = np.load(f"{model_path}/x_grid.npy")
# subs = ['s1', 's2', 's4', 's5', 's6']
# subs = ['s3', 'sAVG']
subs = ['s3']
run_estimation(subs, func_type='gauss', model_path=model_path, n_btrp=1000)
# run_estimation(subs, func_type='sine', model_path=model_path)
# run_estimation(subs, func_type='sine_2peak', model_path=model_path)
# run_estimation(subs, func_type='gauss_2peak_ewk', model_path=model_path)
"""
