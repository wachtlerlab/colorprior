#!/user/bin/env python
# coding=utf-8
"""
Fit PF to estimate 76% correctness threshold of noise level.

x = xShift+sqrt(2)*sd*(erfinv(((yy-chance)/(1-chance-lapse)-.5)*2))

@author: yannansu
@created at: 06.04.21 14:53
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from psychopy import data
from .fit_curves import FitCumNormal
from .load_data import LoadData
# from scipy.optimize import curve_fit


class FitNoise:
    def __init__(self, df):
        self.df = df

    def show_stairs(self):
        grp = self.df.groupby('time_index')
        fig, axes = plt.subplots(grp.ngroups, sharex=True, figsize=(8, 12))

        for i, (t_index, d) in enumerate(grp):
            noise = d.noise.values
            judge = d.judge.values
            wrong_idx = list(np.where(judge == 0)[0])
            correct_idx = list(np.where(judge == 1)[0])

            axes[i].plot(noise, linestyle=':')
            axes[i].plot(noise, linestyle='None', marker='o', fillstyle='none',
                         markersize=5, markevery=wrong_idx)
            axes[i].plot(noise, linestyle='None', marker='o', fillstyle='full',
                         markersize=5, markevery=correct_idx)
            #
            axes[i].set_xlabel('trial')
            axes[i].set_ylim([-25, -5])
            axes[i].set_ylabel('- noise')

            axes[i].label_outer()

        plt.suptitle('Hue angle = ' + str(self.df.standard_stim.unique()))
        fig.tight_layout()

    def fit_noise_pf(self, hue_angle, guess=[-15, 5], bins='unique', lapse=0):

        trial_n = len(self.df)
        intens = self.df[self.df['standard_stim'] == hue_angle]['noise'].values.astype(float)
        reps = self.df[self.df['standard_stim'] == hue_angle]['judge'].values.astype(float)
        bin_intens, bin_reps, bin_N = data.functionFromStaircase(intens, reps, bins=bins)
        sems = [sum(bin_N) / n for n in bin_N]

        chance = 0.5
        fit = FitCumNormal(bin_intens, bin_reps, sems=sems,
                           guess=guess, expectedMin=chance, lapse=lapse)

        fit_params = fit.params
        sigma_params = np.sqrt(np.diagonal(fit.covar))

        CumNormal = lambda xx, xShift, sd: (chance + (1 - chance - lapse) *
                                            ((special.erf((xx - xShift) / (np.sqrt(2) * sd)) + 1) * 0.5))

        xx = np.linspace(-25, 20, 200)

        yy_max = CumNormal(xx, *(fit_params + sigma_params))
        yy_min = CumNormal(xx, *(fit_params - sigma_params))

        thre_y = 0.76
        thre_x = fit.inverse(thre_y)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, r, s in zip(bin_intens, bin_reps, sems):
            ax.plot(i, r, '.', alpha=0.5, markersize=250 / s)

        smoothResp = np.linspace(0.0, 1.0, int(1.0 / .02))
        smoothInt = fit.inverse(smoothResp)
        ax.plot(smoothInt, smoothResp, '-')
        # plt.fill_between(xx, yy_max, yy_min, color='grey', alpha=0.3)

        ax.hlines(y=0.76, xmin=-30, xmax=thre_x, linestyles='dashed', colors='grey')
        ax.vlines(x=fit.inverse(0.76), ymin=0.5, ymax=0.76, linestyles='dashed', colors='grey')

        ssq = np.round(fit.ssq, decimals=3)
        ax.text(3.5, 0.55, 'ssq = ' + str(ssq), fontsize=10)

        ax.set_xlim([-35, 0])
        ax.set_ylim([0., 1.0])
        ax.set_xlabel('- std')
        ax.set_ylabel('correctness')
        plt.title(
            'Hue angle =' + str(hue_angle) + ': threshold = ' + str(np.round(thre_x, 2)) + ', ' + str(trial_n) + ' trials')
        plt.show()

        return thre_x


"""example"""
# df = LoadData('test', data_path='data/noise_test', sel_par=['noise_HH_135']).read_data()
# thre_x = noise_fit_pf(df, hue_angle=135)
