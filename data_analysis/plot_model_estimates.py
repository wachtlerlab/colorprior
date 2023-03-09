#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 03.08.21 14:07

Generate publication-quality figures of modeling estimates.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import pandas as pd
import numpy as np
from data_analysis.color4plot import color4plot
import pickle
from data_analysis.estimate_likelihood import rect_sin

# Figure configuration
plt.style.use('data_analysis/figures/plot_style.txt')
sns.set_context('paper')
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.family'] = 'Arial'
alpha = .6
capsize = 4
fist_x = 22.5
# x_ticks = np.linspace(0 + fist_x, 360 + fist_x, 8, endpoint=False)
x_major_ticks = np.array([0, 90, 180, 270, 360])
x_minor_ticks = np.linspace(0, 360, 8, endpoint=False)
x_grid = np.load("data_analysis/model_estimates/x_grid.npy")


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


def plot_jnd_fits(estm, lh_jnd_fits, save_pdf=None):
    d_l = rep_end(estm.query('condition == "LL"'))
    d_h = rep_end(estm.query('condition == "HH"'))

    hue_angles = d_l['Hue Angle'].values

    jnd_l = d_l['JND'].values
    jnd_h = d_h['JND'].values
    pickle_l, pickle_h = lh_jnd_fits

    with open(pickle_l, 'rb') as f:
        fit_jnd_l = pickle.load(f)
    with open(pickle_h, 'rb') as f:
        fit_jnd_h = pickle.load(f)

    fig, ax = plt.subplots(1, 2, figsize=[3.45 * 2, 3.45])
    ylim = 16
    # LL
    ax[0].set_title('L vs. L')
    ax[0].plot(hue_angles, jnd_l, 'o', markersize=8, color='gray')
    ax[0].errorbar(x=hue_angles, y=jnd_l, yerr=d_l['JND_err'], ls='none', capsize=capsize, color='gray')
    ax[0].plot(x_grid, rect_sin(x_grid, *fit_jnd_l['params']), color='gray')
    ax[0].fill_between(hue_angles, fit_jnd_l['bounds'][0], fit_jnd_l['bounds'][1], alpha=0.2, color='gray')
    ax[0].set_ylim([0, ylim])
    # ax[0].set_ylim([0, 15])

    # HH
    ax[1].set_title('H vs. H')
    ax[1].plot(hue_angles, jnd_h, 'o', markersize=8, color='gray')
    ax[1].errorbar(x=hue_angles, y=jnd_h, yerr=d_h['JND_err'], ls='none', capsize=capsize, color='gray')
    ax[1].plot(x_grid, rect_sin(x_grid, *fit_jnd_h['params']), color='gray')
    ax[1].fill_between(hue_angles, fit_jnd_h['bounds'][0], fit_jnd_h['bounds'][1], alpha=0.2, color='gray')
    # ax[1].set_ylim([0, ylim])
    ax[1].set_ylim([0, 25])

    # Set axis info
    for i in range(2):
        ax[i].set_xlim([0, 360])
        ax[i].xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks))
        ax[i].xaxis.set_major_locator(plt.FixedLocator(x_major_ticks))
        ax[i].set_xlabel('Hue Angle (deg)')
        ax[i].set_ylabel('JND (deg)')

    plt.tight_layout()

    if save_pdf is not None:
        plt.savefig('data_analysis/figures/modeling/' + save_pdf + '.pdf')

    plt.show()


def plot_jnd_fits_both(estm, lh_jnd_fits, save_pdf=None):
    d_l = rep_end(estm.query('condition == "LL"'))
    d_h = rep_end(estm.query('condition == "HH"'))

    hue_angles = d_l['Hue Angle'].values

    jnd_l = d_l['JND'].values
    jnd_h = d_h['JND'].values
    pickle_l, pickle_h = lh_jnd_fits

    with open(pickle_l, 'rb') as f:
        fit_jnd_l = pickle.load(f)
    with open(pickle_h, 'rb') as f:
        fit_jnd_h = pickle.load(f)

    fig, ax = plt.subplots(1, 1, figsize=[3.45, 3.45])
    ylim = 25
    # LL
    ax.set_title('JND fitting')
    ax.scatter(hue_angles, jnd_l, marker='o', s=15, facecolors='none', edgecolors='gray', label='L vs. L')
    ax.errorbar(x=hue_angles, y=jnd_l, yerr=d_l['JND_err'], ls='none', capsize=capsize, color='gray')
    ax.plot(x_grid, rect_sin(x_grid, *fit_jnd_l['params']), color='gray')
    ax.fill_between(hue_angles, fit_jnd_l['bounds'][0], fit_jnd_l['bounds'][1], alpha=0.2, color='gray')

    # HH
    ax.scatter(hue_angles, jnd_h, marker='s', s=15, facecolors='none', edgecolors='gray', label='H vs. H')
    ax.errorbar(x=hue_angles, y=jnd_h, yerr=d_h['JND_err'], ls='none', capsize=capsize, color='gray')
    ax.plot(x_grid, rect_sin(x_grid, *fit_jnd_h['params']), color='gray')
    ax.fill_between(hue_angles, fit_jnd_h['bounds'][0], fit_jnd_h['bounds'][1], alpha=0.2, color='gray')

    ax.set_ylim([0, ylim])
    # ax.set_yscale('log')
    # ax.set_ylim([10**(-0.15), 10**1.5])
    ax.set_xlim([0, 360])
    ax.xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks))
    ax.xaxis.set_major_locator(plt.FixedLocator(x_major_ticks))
    ax.set_xlabel('Hue Angle (deg)')
    ax.set_ylabel('JND (deg)')

    leg = ax.legend(loc=(0.65, 0.85))
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')

    plt.tight_layout()

    if save_pdf is not None:
        plt.savefig('data_analysis/figures/modeling/' + save_pdf + '.pdf')

    plt.show()


def plot_matrix(lh_mat, lh_edges, save_pdf=None):
    """
    :param lh_mat:
    :param lh_edges:
    :param save_pdf:
    :return:
    """
    mat_l, mat_h = lh_mat
    edges_l, edges_h = lh_edges
    # fig, ax = plt.subplots(2, 1, figsize=[3.45, 3.45 * 2 + 0.5])
    fig, ax = plt.subplots(1, 2, figsize=[3.45*2 + 0.5, 3.45])
    extent_l = [edges_l[0][0], edges_l[0][-1], edges_l[1][0], edges_l[1][-1]]
    extent_h = [edges_h[0][0], edges_h[0][-1], edges_h[1][0], edges_h[1][-1]]

    ax[0].set_title('Low-noise')
    im_l = ax[0].imshow(mat_l, origin='lower', cmap='gray', extent=extent_l)
    cb_l = fig.colorbar(im_l, ax=ax[0])
    cb_l.formatter.set_powerlimits((0, 0))  # , shrink=0.65)

    ax[1].set_title('High-noise')
    im_h = ax[1].imshow(mat_h, origin='lower', cmap='gray', extent=extent_h)
    cb_h = fig.colorbar(im_h, ax=ax[1])
    cb_h.formatter.set_powerlimits((0, 0))  # , shrink=0.65)

    # Set axis info
    for i in range(2):
        ax[i].set_xlim([0, 360])
        ax[i].xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks))
        ax[i].xaxis.set_major_locator(plt.FixedLocator(x_major_ticks))
        ax[i].set_xlabel('true stimulus (deg)')

        ax[i].set_ylim([0, 360])
        ax[i].yaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks))
        ax[i].yaxis.set_major_locator(plt.FixedLocator(x_major_ticks))
        ax[i].set_ylabel('measurement (deg)')

    plt.tight_layout()

    if save_pdf is not None:
        plt.savefig('data_analysis/figures/modeling/' + save_pdf + '.pdf')
    plt.show()


"""
################# Running plotting ######################
"""

"""
# For average sub
sub = 'sAVG'
sub_estimates = pd.read_csv('data_analysis/pf_estimates/avg_estimates.csv')
"""

"""
# For a single subject
sub = 's1'
all_estimates = pd.read_csv('data_analysis/pf_estimates/all_estimates.csv')
sub_estimates = all_estimates.query('subject == @sub')
"""

"""
# Make plots
dir = 'data_analysis/model_estimates/'
sub_lh_jnd_fits = [dir + sub + '/' + sub + '_jnd_fit_l.pickle',
                   dir + sub + '/' + sub + '_jnd_fit_h.pickle']
sub_lh_mat = [np.load(dir + sub + '/' + sub + '_mat_l.npy'),
              np.load(dir + sub + '/' + sub + '_mat_h.npy')]
sub_lh_edges = [np.load(dir + sub + '/' + sub + '_mat_edges_l.npy'),
              np.load(dir + sub + '/' + sub + '_mat_edges_h.npy')]

plot_jnd_fits_both(sub_estimates, sub_lh_jnd_fits, save_pdf=sub + '_jnd_fits')
plot_matrix(sub_lh_mat, sub_lh_edges, save_pdf=sub + '_matrix')
"""

"""
# Iterate for all subjects
sub_list = ['s1', 's2', 's3', 's4', 's5', 's6', 'sAVG']
for sub in sub_list:
    if sub == 'sAVG':
        sub_estimates = pd.read_csv('data_analysis/pf_estimates/avg_estimates.csv')
    else:
        all_estimates = pd.read_csv('data_analysis/pf_estimates/all_estimates.csv')
        sub_estimates = all_estimates.query('subject == @sub')

    dir = 'data_analysis/model_estimates/'
    sub_lh_jnd_fits = [dir + sub + '/' + sub + '_jnd_fit_l.pickle',
                       dir + sub + '/' + sub + '_jnd_fit_h.pickle']
    sub_lh_mat = [np.load(dir + sub + '/' + sub + '_mat_l.npy'),
                  np.load(dir + sub + '/' + sub + '_mat_h.npy')]
    sub_lh_edges = [np.load(dir + sub + '/' + sub + '_mat_edges_l.npy'),
                    np.load(dir + sub + '/' + sub + '_mat_edges_h.npy')]

    plot_jnd_fits_both(sub_estimates, sub_lh_jnd_fits, save_pdf=sub + '_jnd_fits')
    plot_matrix(sub_lh_mat, sub_lh_edges, save_pdf=sub + '_matrix')
"""
