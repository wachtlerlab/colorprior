import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_analysis.fit_bstrp import fit_bstrp
import scipy.stats as st
from scipy.special import i0
import json
import pickle
from lmfit import Model


def rep_end(dat):
    """
    Duplicate the first and the last stimuli for visualization.

    """
    first = dat[dat['Hue Angle'] == dat['Hue Angle'].min()]
    first_copy = first.copy()
    first_copy['Hue Angle'] = first_copy['Hue Angle'].apply(lambda x: x + 360)

    last = dat[dat['Hue Angle'] == dat['Hue Angle'].max()]
    last_copy = last.copy()
    last_copy['Hue Angle'] = last_copy['Hue Angle'].apply(lambda x: x - 360)

    rep_dat = pd.concat([dat, first_copy, last_copy],
                        ignore_index=False).sort_values('Hue Angle')
    return rep_dat


def rect_sin(x, a, b):
    """
    Rectified sine function.

    param x:    angle in degree
    param a:    amplitude
    param b:    offset
    """
    shift = -112  # peak at 112
    return a * abs(np.sin((x + shift) * np.pi / 180.0)) + b


def mixed_sin(x, s1, s2, a, b):
    """
    Mixed sine function
    param x:       angle in degree
    param s1, s2:  shifts in degree
    param a:  amplitudes
    param b:  offsets
    """
    return a * (np.sin((x + s1) * np.pi / 90.0) + np.sin((x + s2) * np.pi / 180.0)) + b


def fit_mixed_sin(x, y):
    """
    Fit mixed sine function with lmfit, return predicted values for x grids
    :param x:       x values
    :param y:       y values
    :return:
    """

    model = Model(mixed_sin)

    res = model.fit(y, x=x, s1=45, s2=90, a=1, b=0)

    pars = np.array(list(res.params.valuesdict().values()))

    # stds = np.sqrt(np.diagonal(res.covar))
    # lower = mixed_sin(x_grid, *(pars - stds))
    # upper = mixed_sin(x_grid, *(pars + stds))

    dely = res.eval_uncertainty(x=x_grid, sigma=.68)  # 68% CI
    pred = res.eval(x=x_grid, params=res.params)

    lower = pred - dely
    upper = pred + dely

    fit = {'params': pars,
           'bounds': [lower, upper]}

    return fit


def vonmises_sample(t, jnd, n=100):
    """
    Sample from a customized Von Mises distribution.

    P(m|t) = exp(kappa * cos(t - m)) / (2 * pi * i0(kappa))

    # Note that
    # - The unit of jnd is degree or radian: radian
    # - The relationship between kappa and jnd: kappa = 1/var = 1/(JND**2)

    #Example usage: Test on vonmises_sample function...
    # m = 180
    # jnd = 4.
    # S = vonmises_sample(m, jnd=jnd, n=1000)
    # weights = np.ones_like(S) / len(S)
    # plt.hist(S, bins=10, weights=weights)
    # plt.show()

    :param t      hue angle, in degree, center of the distribution
    :param jnd    kappa = 1/var = 1/(JND**2), the concentration level of the distribution
    :param n      sample size
    """
    # Convert degree to radian
    t = t / 180 * np.pi
    jnd = jnd / 180 * np.pi

    class VonMisesPDF(st.rv_continuous):
        def _pdf(self, m, t, kappa):
            return np.exp(kappa * np.cos(t - m)) / (2 * np.pi * i0(kappa))

    # Par a: Lower bound of the support of the distribution
    # Par b: Upper bound of the support of the distribution
    von_mises = VonMisesPDF(a=0, b=2 * np.pi, name='VonMisesPDF')

    # Determine the kappa value to produce JND values matching the fitted j(θ).
    profile = {'t': t,
               'kappa': 1.0 / (jnd ** 2)}
    if i0(profile['kappa']) == np.inf:
        epsilon_values = np.repeat(t, repeats=n)
    else:
        epsilon_values = von_mises.rvs(t=profile['t'], kappa=profile['kappa'], size=n)

    # Convert the angle from rad to deg
    epsilon_values = epsilon_values / np.pi * 180

    return epsilon_values


def build_matrix(xx, fit):
    """
    Build a matrix: values in x are histogrammed along the 1st dimension (likelihood function),
                    and values in y are histogrammed along the 2nd dimension (measurement distribution).
    :param xx       x gird, should be consistent in the modeling parts
    :param fit      model fitting params of JND (with low- or high-noise data)

    :return:        matrix: 2d array, len(xx) x len(xx)
                    edges:  2d array, [x edges, y edges]

    """
    n_ss = 100
    # ss = []
    # for xi in xx:
    #     # ji = rect_sin(xi, *fit['params'])
    #     ji = mixed_sin(xi, *fit['params'])
    #     si = vonmises_sample(t=xi, jnd=ji, n=n_ss)
    #     ss.append(si)
    # ss = np.array(ss)
    ss = np.array([vonmises_sample(t=xi, jnd=mixed_sin(xi, *fit['params']), n=n_ss) for xi in xx])
    ss_2d = ss.flatten()
    xx_2d = np.repeat(xx, n_ss)

    # The bi-dimensional histogram of samples x and y.
    # Values in x are histogrammed along the 1st dimension and values in y are histogrammed along the 2nd dimension.
    matrix, xedges, yedges = np.histogram2d(x=xx_2d, y=ss_2d, bins=len(xx), normed=True)
    matrix = matrix.T

    xedges = np.round(xedges, 1)
    yedges = np.round(yedges, 1)

    edges = np.array([xedges, yedges])

    return matrix, edges


def estimate_lik(dat, estm, x_grid, n_sample, save_dir):
    """
    Compute and save all estimates before Bayesian inference, which include:
    - fit of JND
    - samples from the measurement distribution
    - matrix of measurement distribution - likelihood function
    - likelihoods
    Note the estimates are always paired for low and high noise conditions.

    :param dat:         data [dataframe]
    :param estm:        estimates [dataframe]
    :param x_grid:      grid for generating the matrices
    :param n_sample:    number of samples for each stimulus pair
    :param save_dir:    data saving path (and partial name), e.g. 'data_analysis/model_estimates/s0/s0'

    :return:            None
    """

    # 1. Load data
    lh_data = dat.query('condition == "LH"')

    unique_pairs = lh_data[['standard_stim', 'test_stim', 'actual_intensity']]. \
        drop_duplicates().sort_values(by=['standard_stim', 'test_stim'])

    stim_l = unique_pairs['standard_stim'].values.astype('float')
    stim_h = unique_pairs['test_stim'].values.astype('float')

    # 2. Fit JND
    d_l = rep_end(estm.query('condition == "LL"'))
    d_h = rep_end(estm.query('condition == "HH"'))

    hue_angles = d_l['Hue Angle'].values

    jnd_l = d_l['JND'].values
    jnd_h = d_h['JND'].values

    # # LL
    # fit_l = fit_bstrp(hue_angles, jnd_l, func=rect_sin, n_bs=1000, p_ci=.68)  #, p0=[1., 1, -112.5])
    # jnd_stim_l = rect_sin(stim_l, *fit_l['params'])
    # # HH
    # fit_h = fit_bstrp(hue_angles, jnd_h, func=rect_sin, n_bs=1000, p_ci=.68)  #, p0=[1., 1, -112.5])
    # jnd_stim_h = rect_sin(stim_h, *fit_h['params'])

    # LL
    # fit_l = fit_bstrp(hue_angles, jnd_l, func=mixed_sin, n_bs=1000, p_ci=.68, p0=[45, 90, 1, 0])
    fit_l = fit_mixed_sin(hue_angles, jnd_l)
    jnd_stim_l = mixed_sin(stim_l, *fit_l['params'])
    # HH
    # fit_h = fit_bstrp(hue_angles, jnd_h, func=mixed_sin, n_bs=1000, p_ci=.68, p0=[45, 90, 1, 0])
    fit_h = fit_mixed_sin(hue_angles, jnd_h)
    jnd_stim_h = mixed_sin(stim_h, *fit_h['params'])

    # same for LH... but not used for matrix
    d_lh = rep_end(estm.query('condition == "LH"'))
    hue_angles = d_lh['Hue Angle'].values
    jnd_lh = d_lh['JND'].values
    # fit_lh = fit_bstrp(hue_angles, jnd_lh, func=rect_sin, n_bs=1000, p_ci=.68)  # , p0=[1., 1, -112.5])
    # fit_lh = fit_bstrp(hue_angles, jnd_lh, func=mixed_sin, n_bs=1000, p_ci=.68, p0=[45, 90, 1, 0])
    fit_lh = fit_mixed_sin(hue_angles, jnd_lh)


    # """
    # 3. Build matrix
    mat_l, edges_l = build_matrix(x_grid, fit_l)
    mat_h, edges_h = build_matrix(x_grid, fit_h)
    #
    # 4. Sample and estimate likelihoods
    # samples:      2d array, len(unique stimulus pairs) x n_sample
    # likelihoods:  3d array, len(unique stimulus pairs) x n_sample x len(x_grid)
    #
    samples_l = np.sort(np.array([vonmises_sample(t=s, jnd=j, n=n_sample)
                                  for s, j in zip(stim_l, jnd_stim_l)]), axis=1)
    samples_h = np.sort(np.array([vonmises_sample(t=s, jnd=j, n=n_sample)
                                  for s, j in zip(stim_h, jnd_stim_h)]), axis=1)
    samples_size = np.shape(samples_l)

    likelihoods_l = np.array([mat_l[np.abs(x_grid - sp).argmin(), :] for sp in samples_l.flatten()])
    likelihoods_l = np.reshape(likelihoods_l, (samples_size[0], samples_size[1], len(x_grid)))

    likelihoods_h = np.array([mat_h[np.abs(x_grid - sp).argmin(), :] for sp in samples_h.flatten()])
    likelihoods_h = np.reshape(likelihoods_h, (samples_size[0], samples_size[1], len(x_grid)))
    # """
    # 5. Save data
    with open(save_dir + '_jnd_fit_l.pickle', 'wb') as f_l:
        pickle.dump(fit_l, f_l)
    with open(save_dir + '_jnd_fit_h.pickle', 'wb') as f_h:
        pickle.dump(fit_h, f_h)
    with open(save_dir + '_jnd_fit_lh.pickle', 'wb') as f_lh:
        pickle.dump(fit_lh, f_lh)
    # """
    np.save(save_dir + '_mat_l.npy', mat_l)
    np.save(save_dir + '_mat_h.npy', mat_h)
    np.save(save_dir + '_mat_edges_l.npy', edges_l)
    np.save(save_dir + '_mat_edges_h.npy', edges_h)

    np.save(save_dir + '_samples_l.npy', samples_l)
    np.save(save_dir + '_samples_h.npy', samples_h)
    np.save(save_dir + '_likelihoods_l.npy', likelihoods_l)
    np.save(save_dir + '_likelihoods_h.npy', likelihoods_h)
    # """


"""
################# Running estimation ######################
"""

# """
model_path = "data_analysis/model_estimates_v2"

# x_grid = np.arange(0.0001, 359.9999, 2.)
x_grid = np.load(f"{model_path}/x_grid.npy")
n_sample = 100
# """

"""
# For single sub
all_data = pd.read_csv('data/all_sel_data.csv')
all_estimates = pd.read_csv('data_analysis/pf_estimates/all_estimates.csv')

# sub_list = ['s1', 's2', 's4', 's5', 's6']

# for sub in sub_list:
for sub in ['s3']:
    sub_in_data = sub[0] + '0' + sub[1]                  # different naming way in data
    sub_data = all_data.query('subject == @sub_in_data')
    sub_estimates = all_estimates.query('subject == @sub')
    sub_path = f"{model_path}/{sub}/{sub}"
    estimate_lik(sub_data, sub_estimates, x_grid, n_sample, sub_path)
"""

"""
# For average sub
all_data = pd.read_csv('data/all_sel_data.csv')
avg_estimates = pd.read_csv('data_analysis/pf_estimates/avg_estimates.csv')
sub_path = f"{model_path}/sAVG/sAVG"
estimate_lik(all_data, avg_estimates, x_grid, n_sample, sub_path)
"""