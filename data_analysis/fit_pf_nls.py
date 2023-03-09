#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 22.03.21 18:10

Fit Psychometric functions using non-linear least squares.

Example usage:
test_data = LoadData('s01', data_path='data', sel_par=['LL_2x2']).read_data()
test_FitPf = FitPf(test_data)    # or test_FitPf = FitPf_Correctness(test_data)
test_fit = test_FitPf.fit()
test_FitPf.plot_pf_curve()
test_FitPf.plot_pf_param()       # not available for FitPf_Correctness

"""

import pandas as pd
import numpy as np
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
from psychopy import data
from .fit_curves import FitCumNormal
from .color4plot import color4plot
from matplotlib import rc

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc("font", **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans'], 'size': 15})

# Set the font used for MathJax - more on this later
rc('mathtext', **{"default": 'regular'})

# Define params for plotting
plot_config = {"figsize": (16, 10),
               'title_fontsize': 13,
               'label_fontsize': 12,
               'tick_size': 8
               }


# mpl.rcParams['agg.path.chunksize'] = 1000000000


class FitPf:
    """
    Fit 2AFC data to a psychometric curve (response 'Yes' - stimulus intensity)

    Example usage:
        df = LoadData('test', sel_par=['LL_set1']).read_data()  # load data
        fit_dat = FitPf(df).fit()  # do the fit
        FitPf(df).plot_pf_curve()  # plot PF curves
        FitPf(df).plot_pf_param()  # plot PF estimates

    """

    def __init__(self, df, guess=None, lapse=0.0, bins='unique'):
        """
        :param df:      psychometric data (Dataframe)
        :param guess:   guess parameter for fitting, default None as [0., 5.] (corresponding to [PSE, JND])
        :param lapse:   lapse rate, default as 0.
        :param bins:    number of bins for binning data before fitting, default as 'unique'
        """
        self.df = df
        if guess is None:
            self.guess = [0., 5.]
        else:
            self.guess = guess
        self.lapse = lapse
        self.bins = bins

    def fit(self):
        df_dict = dict(list(self.df.groupby('standard_stim')))
        # A nested dictionary
        fit_dat = {}

        for key, d in df_dict.items():
            fit_dat[key] = d.to_dict(orient='list')

            fit_dat[key]['Hue Angle'] = key
            fit_dat[key]['Trial N'] = len(df_dict[key])

            # Bin data
            fit_dat[key]['Binned Intensities'], \
            fit_dat[key]['Binned Responses'], \
            fit_dat[key]['Binned N'] = data.functionFromStaircase(
                fit_dat[key]['actual_intensity'],
                fit_dat[key]['resp_as_larger'],
                bins=self.bins)
            # Sems is defined as 1/weight in Psychopy
            fit_dat[key]['sems'] = [sum(fit_dat[key]['Binned N']) / n
                                    for n in fit_dat[key]['Binned N']]

            fit_dat[key]['fit'] = FitCumNormal(fit_dat[key]['Binned Intensities'],
                                               fit_dat[key]['Binned Responses'],
                                               sems=fit_dat[key]['sems'], guess=self.guess,
                                               expectedMin=0.0, lapse=self.lapse)  # customized cumulative Gaussian

            fit_dat[key]['PSE'], fit_dat[key]['JND'] = fit_dat[key]['fit'].params
            fit_dat[key]['PSE_err'], fit_dat[key]['JND_err'] = np.sqrt(np.diagonal(fit_dat[key]['fit'].covar))
            fit_dat[key]['ssq'] = fit_dat[key]['fit'].ssq

        return fit_dat

    def plot_pf_curve(self, save_pdf=None):
        """
        Plot PFs from nonlinear least squares fitting results.

        """
        fit_dat = self.fit()
        num = len(fit_dat)

        if num == 1:
            fig, axes = plt.subplots(num, 1, figsize=plot_config['figsize'])
        else:
            fig, axes = plt.subplots(2, int(num / 2), figsize=plot_config['figsize'])

        xlim = [-20, 20]
        ylim = [0, 1.0]

        for idx, key in enumerate(fit_dat):

            this_dat = fit_dat[key]
            this_fit = this_dat['fit']
            ntrial = this_dat['Trial N']

            if num == 1:
                ax = axes
            else:
                ax = axes.flatten()[idx]

            hue_angle = this_dat['Hue Angle']
            color_code = color4plot(hue_angle)[0]

            # hue_angles = np.array([float(k) for k in fit_dat.keys()])
            # color = colorcodes[int((key - self.first_angle) / self.first_angle / 2)]
            for inten, resp, se in zip(this_dat['Binned Intensities'],
                                       this_dat['Binned Responses'],
                                       this_dat['sems']):
                ax.plot(inten, resp, '.', color=color_code, alpha=0.5, markersize=30 / np.log(se))

            smoothResp = pylab.arange(0.0, 1.0, .02)
            smoothInt = this_fit.inverse(smoothResp)
            ax.plot(smoothInt, smoothResp, '-', color=color_code)  # plot fitted curve

            for val in [0.25, 0.5, 0.75]:
                ax.hlines(y=val, xmin=xlim[0], xmax=this_fit.inverse(val), linestyles='dashed', colors='grey')
                ax.vlines(x=this_fit.inverse(val), ymin=ylim[0], ymax=val, linestyles='dashed', colors='grey')

            ssq = np.round(this_dat['ssq'], decimals=3)  # sum-squared error
            ax.text(3.5, 0.55, 'ssq = ' + str(ssq), fontsize=plot_config['tick_size'])
            ax.set_title('hue_angle: ' + str(hue_angle) + ', ' + '%dtrials' % ntrial,
                         fontsize=plot_config['title_fontsize'])
            ax.set_xlim(xlim)
            # ax.set_ylim(ylim)
            ax.tick_params(axis='both', which='major', labelsize=plot_config['tick_size'])

        if num == 1:
            x_ax = ax
            y_ax = ax
        elif num == 2:
            x_ax = axes[-1]
            y_ax = axes[0]
        else:
            x_ax = axes[-1, :]
            y_ax = axes[:, 0]

        plt.setp(x_ax, xlabel='Hue Angle')
        plt.setp(y_ax, ylabel='Response "Test hue angle is larger" ')

        plt.setp(ax.get_xticklabels(), fontsize=plot_config['tick_size'])
        plt.setp(ax.get_yticklabels(), fontsize=plot_config['tick_size'])

        # fig.suptitle(self.sub[0:2] + '_' + str(ntrial) + 'trials', fontsize=plot_config['title_fontsize'])
        fig.suptitle(str(ntrial) + ' trials', fontsize=plot_config['title_fontsize'])
        if save_pdf is not None:
            plt.savefig('data_analysis/figures/' + save_pdf + '.pdf')
        plt.show()

    def plot_pf_param(self, save_pdf=None):
        """
        Plot estimated PF parameters from nonlinear least squares fitting results.

        """

        fit_dat = pd.DataFrame(self.fit()).T
        num = len(fit_dat)
        hue_angles = fit_dat.index
        color_codes = color4plot(hue_angles)
        plt.figure(figsize=(6, 5))
        plt.title('Cumulative Gaussian Parameter Etimates, ' + '%dtrials' % fit_dat['Trial N'].unique()[0],
                  fontsize=plot_config['title_fontsize'])
        plt.xlabel('Hue Angle', fontsize=plot_config['label_fontsize'])
        plt.ylabel('Parameter Estimates', fontsize=plot_config['label_fontsize'])

        ax = plt.subplot(111)
        ax.errorbar(hue_angles, fit_dat.PSE, yerr=fit_dat.PSE_err, label='PSE', ls='-', color=[0.3, 0.3, 0.3])
        ax.errorbar(hue_angles, fit_dat.JND, yerr=fit_dat.JND_err, label='JND', ls='-', color=[0.6, 0.6, 0.6])
        ax.scatter(hue_angles, fit_dat.PSE, color=color_codes, s=60)
        ax.scatter(hue_angles, fit_dat.JND, color=color_codes, s=60)

        ax.hlines(0, 0, 360, linestyles='dashed', color='silver')

        # xlabels = [f"{l}\n{a}" for l, a in zip(psypar['hue_id'], psypar['angle'])]
        # xlabels = hue_angles
        ax.set_xticks(hue_angles)
        ax.set_xticklabels(hue_angles, rotation=45, fontsize=plot_config['tick_size'])
        ax.set_xlim([0, 360])

        plt.legend(fontsize=plot_config['tick_size'])
        if save_pdf is not None:
            plt.savefig('data_analysis/figures/' + save_pdf + '.pdf')
        plt.show()


class FitPf_Correctness:
    """
    Fit 2AFC data to a psychometric curve (response correctness - (absolute) stimulus intensity)
    Note it is similar to the class FitPF, but with different Y values.
    """

    def __init__(self, df, guess=None, lapse=0.0, bins=None, func='CumNormal'):
        """
        :param df:      psychometric data (Dataframe)
        :param guess:   guess parameter for fitting, default None as [0., 5.] (corresponding to [PSE, JND])
        :param lapse:   lapse rate, default as 0.
        :param bins:    number of bins for binning data before fitting, default as 'unique'
        """
        self.df = df
        self.df['labeled_stim'] = (-1) ** (self.df['actual_intensity'] < 0) * self.df[
            'standard_stim']  # to make the following fitting easier, create a new column of standard stimulus with sign labels
        if guess is None:
            self.guess = [5., 1.]
        else:
            self.guess = guess
        self.lapse = lapse
        if bins is None:
            self.bins = 'unique'
        else:
            self.bins = bins
        self.func = func

    def fit(self):
        df_dict = dict(list(self.df.groupby(['standard_stim', 'labeled_stim'])))
        # A nested dictionary
        fit_dat = {}

        for key, d in df_dict.items():
            fit_dat[key] = d.to_dict(orient='list')
            fit_dat[key]['Hue Angle'] = key[0]
            fit_dat[key]['labeled_stim'] = key[1]
            fit_dat[key]['Trial N'] = len(df_dict[key])
            fit_dat[key]['actual_intensity'] = [abs(x) for x in fit_dat[key]['actual_intensity']]

            # Bin data
            fit_dat[key]['Binned Intensities'], \
            fit_dat[key]['Binned Responses'], \
            fit_dat[key]['Binned N'] = data.functionFromStaircase(
                (fit_dat[key]['actual_intensity']),
                fit_dat[key]['judge'],
                bins=self.bins)

            # Sems is defined as 1/weight in Psychopy
            fit_dat[key]['sems'] = [sum(fit_dat[key]['Binned N']) / n
                                    for n in fit_dat[key]['Binned N']]

            if self.func == 'CumNormal':
                # customized cumulative Gaussian
                fit_dat[key]['fit'] = FitCumNormal(fit_dat[key]['Binned Intensities'],
                                                   fit_dat[key]['Binned Responses'],
                                                   sems=fit_dat[key]['sems'], guess=self.guess,
                                                   expectedMin=0.5, lapse=self.lapse)
            elif self.func == 'Weibull':
                fit_dat[key]['fit'] = data.FitWeibull(fit_dat[key]['Binned Intensities'],
                                                      fit_dat[key]['Binned Responses'],
                                                      sems=fit_dat[key]['sems'], guess=self.guess,
                                                      expectedMin=0.5)
            elif self.func == 'Logistic':
                fit_dat[key]['fit'] = data.FitLogistic(fit_dat[key]['Binned Intensities'],
                                                       fit_dat[key]['Binned Responses'],
                                                       sems=fit_dat[key]['sems'], guess=self.guess,
                                                       expectedMin=0.5)
            else:
                raise ValueError("Given fitting function is not found.")

            fit_dat[key]['PSE'], fit_dat[key]['JND'] = fit_dat[key]['fit'].params
            fit_dat[key]['PSE_err'], fit_dat[key]['JND_err'] = np.sqrt(np.diagonal(fit_dat[key]['fit'].covar))
            fit_dat[key]['ssq'] = fit_dat[key]['fit'].ssq

        return fit_dat

    def plot_pf_curve(self):
        """
        Plot PFs from nonlinear least squares fitting results.

        """
        fit_dat = self.fit()
        num = len(fit_dat)

        if num == 1:
            fig, axes = plt.subplots(num, 1, figsize=plot_config['figsize'])
        else:
            fig, axes = plt.subplots(2, int(num / 4), figsize=plot_config['figsize'])

        xlim = [0, 12]
        ylim = [.5, 1.]

        for idx, key in enumerate(fit_dat):

            this_dat = fit_dat[key]
            this_fit = this_dat['fit']
            ntrial = this_dat['Trial N']

            if num == 1:
                ax = axes
            else:
                ax = axes.flatten()[int(np.floor(idx / 2))]

            if key[1] < 0:
                label = 'minus'
                marker = 'o'
                color = 'coral'
            else:
                label = 'plus'
                marker = 'P'
                color = 'skyblue'
            hue_angle = this_dat['Hue Angle']
            # color_code = color4plot(hue_angle)[0]

            # hue_angles = np.array([float(k) for k in fit_dat.keys()])
            # color = colorcodes[int((key - self.first_angle) / self.first_angle / 2)]
            for inten, resp, se in zip(this_dat['Binned Intensities'],
                                       this_dat['Binned Responses'],
                                       this_dat['sems']):
                ax.plot(inten, resp, '.', color=color, alpha=.5, markersize=5 / np.log10(se), marker=marker)

            smoothInt = pylab.arange(0, 12, .5)
            smoothResp = this_fit.eval(smoothInt)
            ax.plot(smoothInt, smoothResp, '--', color=color, label=label)  # plot fitted curve

            # ax.hlines(y=0.75, xmin=0, xmax=this_fit.inverse(0.75), linestyles='dashed', colors='grey')
            # ax.vlines(x=this_fit.inverse(0.75), ymin=0.5, ymax=0.75, linestyles='dashed', colors='grey')

            # ssq = np.round(this_dat['ssq'], decimals=3)  # sum-squared error
            # ax.text(3.5, 0.55, 'ssq = ' + str(ssq), fontsize=plot_config['tick_size'])
            ax.set_title('hue_angle: ' + str(hue_angle) + ', ' + '%dtrials' % ntrial,
                         fontsize=plot_config['title_fontsize'])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(axis='both', which='major', labelsize=plot_config['tick_size'])

        if num == 1:
            x_ax = ax
            y_ax = ax
        elif num == 2:
            x_ax = axes[-1]
            y_ax = axes[0]
        else:
            x_ax = axes[-1, :]
            y_ax = axes[:, 0]

        plt.setp(x_ax, xlabel='Hue Angle')
        plt.setp(y_ax, ylabel='Correct Response')

        plt.setp(ax.get_xticklabels(), fontsize=plot_config['tick_size'])
        plt.setp(ax.get_yticklabels(), fontsize=plot_config['tick_size'])

        # fig.suptitle(self.sub[0:2] + '_' + str(ntrial) + 'trials', fontsize=plot_config['title_fontsize'])
        fig.suptitle(str(ntrial) + ' trials', fontsize=plot_config['title_fontsize'])
        plt.legend()
        plt.show()


# s05_lh = LoadData('s05', data_path='data', sel_par=['LH_2x2']).read_data()
# # test_fit = FitPf_Correctness(s05_lh, bins=6).fit()
# # FitPf(s05_lh, guess=[1, 2]).plot_pf_curve()
# FitPf_Correctness(s05_lh, guess=[5, 5], func='CumNormal').plot_pf_curve()
