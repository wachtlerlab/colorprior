#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 25.06.21 11:18

Generate publication-quality figures of psychometric function estimates.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import pandas as pd
import numpy as np
from data_analysis.color4plot import color4plot

# Figure configuration
# sns.set_context("poster")
plt.style.use('data_analysis/figures/plot_style.txt')
# plt.style.use('data_analysis/figures/plot_style_poster.txt')

alpha = {'line': 1., 'marker': 0.7}
capsize = 4
fist_x = 22.5
# x_ticks = np.linspace(0 + fist_x, 360 + fist_x, 8, endpoint=False)
x_major_ticks = np.array([0, 90, 180, 270, 360])
x_minor_ticks = np.linspace(0, 360, 8, endpoint=False)
legend_loc = [0.75, 0.9]
gridline = {'color': [.8, .8, .8], 'alpha': 0.4, 'width': 1.5}


def rep_end(estimates):
    """
    Duplicate the first and the last stimuli for visualization.

    """
    first = estimates[estimates['Hue Angle'] == estimates['Hue Angle'].min()]
    first_copy = first.copy()
    first_copy['Hue Angle'] = first_copy['Hue Angle'].apply(lambda x: x + 360)

    last = estimates[estimates['Hue Angle'] == estimates['Hue Angle'].max()]
    last_copy = last.copy()
    last_copy['Hue Angle'] = last_copy['Hue Angle'].apply(lambda x: x - 360)

    estimates_plt = pd.concat([estimates, first_copy, last_copy], ignore_index=False).sort_values('Hue Angle')
    return estimates_plt


# JND
def plot_all_JND_by_condition(estimates, save_pdf=None):
    cond_list = ['LL', 'HH', 'LH']
    title_list = ['L vs. L', 'H vs. H', 'L vs. H']
    # cond_list = estimates['condition'].unique()
    cond_num = len(cond_list)
    fig, axes = plt.subplots(nrows=1, ncols=cond_num, figsize=[10, 5])
    # plt.suptitle('Discrimination Thresholds Across Conditions')

    sub_list = np.sort(estimates['subject'].unique())
    sub_num = len(sub_list)
    l_gray = np.repeat(np.linspace(.2, .8, sub_num, endpoint=True), 3).reshape(-1, 3)
    l_color = {s: g for s, g in zip(sub_list, l_gray)}
    dt_color = color4plot(estimates['Hue Angle'].unique())
    markers = ['o', '^', 'v', 'X', 's', '+', 'd', '*']
    df_marker = {s: m for s, m in zip(sub_list, markers[0:sub_num])}

    for idx, cond in enumerate(cond_list):
        axes[idx].set_title(title_list[idx])
        for key, grp in estimates[estimates.condition == cond].groupby(['subject']):
            axes[idx].errorbar(x=grp['Hue Angle'], y=grp['JND'], yerr=grp['JND_err'],
                               c=l_color[key], linestyle='dashed', capsize=capsize)
            axes[idx].scatter(x=grp['Hue Angle'], y=grp['JND'],
                              c=dt_color, marker=df_marker[key], edgecolors='none', label=key)

        axes[idx].set_xlabel('Hue Angle (deg)')
        axes[idx].set_xticks(x_major_ticks)
        # axes[idx].set_xlim([-fist_x, 360 + fist_x])
        axes[idx].set_xlim([0, 360])
        axes[idx].set_xticklabels(x_major_ticks, rotation=45)

        ylim = 25
        axes[idx].set_ylim([0, ylim])
        # [plt.vlines(x, 0, ylim, colors='grey', linestyles='-', alpha=0.5) for x in [0, 360, 112.5, 112.5 + 180]]

    leg = axes[0].legend()
    [marker.set_color([.66, .66, .66]) for marker in leg.legendHandles]
    axes[0].set_ylabel('JND (deg)')

    plt.tight_layout()
    if save_pdf is not None:
        plt.savefig('data_analysis/figures/PF_estimates' + save_pdf + '.pdf')
    plt.show()


def plot_all_PSE_lh(estimates, save_pdf=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    # plt.title('Relative Bias (L vs. H)')

    sub_list = np.sort(estimates['subject'].unique())
    sub_num = len(sub_list)
    l_gray = np.repeat(np.linspace(.2, .8, sub_num, endpoint=True), 3).reshape(-1, 3)
    l_color = {s: g for s, g in zip(sub_list, l_gray)}
    dt_color = color4plot(estimates['Hue Angle'].unique())
    markers = ['o', '^', 'v', 'X', 's', '+', 'd', '*']
    df_marker = {s: m for s, m in zip(sub_list, markers[0:sub_num])}
    dt_size = 50

    for key, grp in estimates[estimates.condition == 'LH'].groupby(['subject']):
        ax.errorbar(x=grp['Hue Angle'], y=grp['PSE'], yerr=grp['PSE_err'],
                    c=l_color[key], linestyle='dashed', capsize=capsize)
        ax.scatter(x=grp['Hue Angle'], y=grp['PSE'],
                   c=dt_color, s=dt_size, marker=df_marker[key], edgecolors='none', label=key)

    leg = ax.legend()
    [marker.set_color([.66, .66, .66]) for marker in leg.legendHandles]

    plt.xlabel('Hue Angle (deg)')
    fist_x = 22.5
    x_ticks = np.linspace(0 + fist_x, 360 + fist_x, 8, endpoint=False)
    plt.xticks(x_ticks)
    plt.xlim([0, 360])
    # plt.hlines(0, 0, 360, colors='grey', alpha=0.5)

    plt.ylabel('PSE (deg)')
    ylim_abs = 12
    plt.ylim([-ylim_abs, ylim_abs])
    # [plt.vlines(x, -ylim_abs, ylim_abs, colors='grey', linestyles='-', alpha=0.5) for x in [0, 360, 112.5, 112.5 + 180]]
    if save_pdf is not None:
        plt.savefig('data_analysis/figures/PF_estimates' + save_pdf + '.pdf')
    plt.show()


def plot_sub_estimates(estimates, save_pdf=None):
    labels = {'LL': 'L vs. L',
              'HH': 'H vs. H',
              'LH': 'L vs. H'}
    linecolors = {'LL': [.3, .3, .3],
                  'HH': [.7, .7, .7],
                  'LH': [.5, .5, .5]}
    markers = {'LL': 'o',
               'HH': 's',
               'LH': 'v'}
    marker_color = color4plot(estimates['Hue Angle'].unique())

    fig, axes = plt.subplots(figsize=(3.5, 10), nrows=3, ncols=1, sharex='none', sharey='none')

    # Plot JND, same-noise
    # [axes[0].vlines(x, 0, 15, alpha=gridline['alpha'], colors=gridline['color'], linewidth=gridline['width'], zorder=1)
    # for x in x_major_ticks]
    for key, grp in estimates.query("condition=='LL' or condition=='HH'").groupby('condition'):
        axes[0].errorbar(x=grp['Hue Angle'], y=grp['JND'], yerr=grp['JND_err'],
                         c=linecolors[key], linestyle='dashed', capsize=capsize, zorder=2)
        axes[0].scatter(x=grp['Hue Angle'], y=grp['JND'],
                        c=marker_color, marker=markers[key], edgecolors='none', alpha=alpha['marker'],
                        label=labels[key], zorder=3)
    # ax[0].set_yscale('log')a
    # ax[0].set_ylim([10**(0), 10**1.5])
    # axes[0].set_ylim([0, 25])
    # axes[0].set_yticks([0, 5, 10, 15, 20, 25])
    axes[0].set_ylim([0, 15])   # For s5 and sAVG, use smaller scale
    axes[0].set_yticks([0, 5, 10, 15])
    axes[0].set_ylabel('JND (deg), same-noise')

    axes[0].set_xlim([0, 360])
    axes[0].xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks))
    axes[0].xaxis.set_major_locator(plt.FixedLocator(x_major_ticks))

    leg = axes[0].legend(loc=legend_loc)
    # leg.get_frame().set_linewidth(0.0)
    # leg.get_frame().set_facecolor('none')
    # [marker.set_color([.66, .66, .66]) for marker in leg.legendHandles]

    # Plot JND, cross-noise
    # [axes[1].vlines(x, 0, 15, alpha=gridline['alpha'], colors=gridline['color'], linewidth=gridline['width'], zorder=1)
    #  for x in x_major_ticks]
    estimates_LH = estimates.query("condition=='LH'")
    axes[1].errorbar(x=estimates_LH['Hue Angle'], y=estimates_LH['JND'], yerr=estimates_LH['JND_err'],
                     c=linecolors['LH'], linestyle='dashed', capsize=capsize, zorder=2)
    axes[1].scatter(x=estimates_LH['Hue Angle'], y=estimates_LH['JND'],
                    c=marker_color, marker=markers['LH'], edgecolors='none', alpha=alpha['marker'],
                    label='L vs. H', zorder=3)
    # ax[1].set_yscale('log')
    # ax[1].set_ylim([10**(-1), 10**1.5])
    axes[1].set_ylim([0, 15])
    axes[1].set_yticks([0, 5, 10, 15])
    axes[1].set_ylabel('JND (deg), cross-noise')

    axes[1].set_xlim([0, 360])
    axes[1].xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks))
    axes[1].xaxis.set_major_locator(plt.FixedLocator(x_major_ticks))
    # plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)
    axes[1].set_xlabel('Hue Angle (deg)')

    leg = axes[1].legend(loc=legend_loc)
    # leg.get_frame().set_linewidth(0.0)
    # leg.get_frame().set_facecolor('none')
    # [marker.set_color([.66, .66, .66]) for marker in leg.legendHandles]

    # Plot PSE, cross-noise
    # [axes[2].vlines(x, -10, 10,
    #                 alpha=gridline['alpha'], colors=gridline['color'], linewidth=gridline['width'], zorder=1)
    #  for x in x_major_ticks]
    axes[2].errorbar(x=estimates_LH['Hue Angle'], y=estimates_LH['PSE'], yerr=estimates_LH['PSE_err'],
                     c=linecolors['LH'], linestyle='dashed', capsize=capsize, zorder=2)
    axes[2].scatter(x=estimates_LH['Hue Angle'], y=estimates_LH['PSE'],
                    c=marker_color, marker=markers['LH'], edgecolors='none', alpha=alpha['marker'],
                    label='L vs. H', zorder=3)

    # axes[2].hlines(0, 0, 360, alpha=gridline['alpha'],  colors=gridline['color'], linewidth=gridline['width'],zorder=1)
    # axes[2].set_ylim([-15, 15])
    axes[2].set_ylim([-10, 10])   # For s5 and sAVG, use smaller scale
    axes[2].set_yticks([-15, -10, -5, 0, 5, 10, 15])

    axes[2].set_ylabel('PSE (deg), cross-noise')

    axes[2].set_xlim([0, 360])
    axes[2].xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks))
    axes[2].xaxis.set_major_locator(plt.FixedLocator(x_major_ticks))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=0)
    # axes[2].set_xticks(x_ticks)
    # axes[2].set_xticklabels(x_ticks, rotation=45)
    axes[2].set_xlabel('Hue Angle (deg)')

    plt.tight_layout()
    if save_pdf is not None:
        plt.savefig('data_analysis/figures/PF_estimates/' + save_pdf + '.pdf')
    plt.show()


# ================================== Generate plots ================================

"""
# 1. For single subject's estimates
# 1.1. Load data
all_estimates = pd.read_csv('data_analysis/pf_estimates/all_estimates.csv')
all_estimates = all_estimates.query("subject == 's5'")
# 1.2. Plot pooled data
plot_all_JND_by_condition(rep_end(all_estimates), save_pdf='pooled_JND_by_cond')
plot_all_PSE_lh(rep_end(all_estimates), save_pdf='pooled_PSE_lh')
# 1.3. Plot estimates for each subject
for key, grp in all_estimates.groupby('subject'):
    plot_sub_estimates(rep_end(grp), save_pdf='PF_estimates' + '_' + key)
"""
"""
# 2.For average data/estimates
# 2.1. Average subject: estimates from pooled data
avg_estimates = pd.read_csv('data_analysis/pf_estimates/avg_estimates.csv')
plot_all_PSE_lh(rep_end(avg_estimates), save_pdf='sAvg_PSE_lh')
plot_sub_estimates(rep_end(avg_estimates), save_pdf='PF_estimates_sAvg')
"""
"""
# 2.2 Average of estimates
simple_avg_estimates = pd.read_csv('data_analysis/pf_estimates/simple_avg_estimates.csv')
plot_sub_estimates(rep_end(simple_avg_estimates), save_pdf='PF_estimates_simple_avg')
"""