#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 14.06.21 11:51

Fit Psychometric functions using Bayesfit.

Example usage:
test_data = LoadData('s01', data_path='data', sel_par=['LL_2x2']).read_data()
test_FitPf = FitPf_bayes(test_data, ylabel='correct', func='norm', params=[[None, None], [True, True, False, False]])
fit_dict = test_FitPf.fit()
fit_df = test_FitPf._to_df()
test_FitPf.plot_pf_curve()

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_analysis.color4plot import color4plot
import bayesfit as bf
from bayesfit.psyFunction import psyfunction as _psyfunction
from matplotlib import rc
from data_analysis.load_data import LoadData

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


class FitPf_bayes:
    def __init__(self, df, ylabel="response", func='norm', params=None):
        self.df = df
        self.df['labeled_stim'] = (-1) ** (self.df['actual_intensity'] < 0) * self.df[
            'standard_stim']  # to make the following fitting easier, create a new column of standard stimulus with sign labels
        self.ylabel = ylabel
        if self.ylabel not in ['response', 'correct']:
            raise ValueError("Given ylabel dose not apply!")
        self.func = func
        if self.func not in ["norm", "logistic", "weibull"]:
            raise ValueError("Given fitting function type dose not apply!")
        self.params = params
        if self.params is None:
            self.params = [[None, None],
                           [True, True, False, False]]
    def fit(self):
        """
        Fit PF to data using Bayes fitting.
        :return:
        """
        if self.ylabel == 'response':
            grp_keys = ['standard_stim']
        elif self.ylabel == 'correct':
            grp_keys = ['standard_stim', 'labeled_stim']
        df_dict = dict(list(self.df.groupby(grp_keys)))
        # fit_dict = pd.DataFrame(columns=['scale', 'slope', 'threshold'])
        fit_dict = {}
        for key, df in df_dict.items():
            fit_dict[key] = {}
            fit_dict[key]['ylabel'] = self.ylabel
            fit_dict[key]['ntrial'] = len(df)
            if self.ylabel == 'response':
                fit_dict[key]['hue_angle'] = key
                data = df.groupby('actual_intensity')['resp_as_larger'].agg(['sum', 'count'])
                chance = 0.
                threshold = 0.5
                intensity = data.index.values
            else:
                fit_dict[key]['hue_angle'] = key[0]
                fit_dict[key]['labeled_stim'] = key[1]
                data = df.groupby('actual_intensity')['judge'].agg(['sum', 'count'])
                intensity = abs(data.index.values)
                chance = 0.5
                threshold = 0.75
            # Reform data as a m-row by 3-column Numpy array:
            # Stimulus intensity	N-trials correct	N-trials total
            data_matrix = np.transpose([intensity,
                                        data['sum'].values,
                                        data['count'].values])

            metrics, options = bf.fitmodel(data_matrix, nafc=2, sigmoid_type=self.func,
                                           threshold=threshold, density=100,
                                           param_ests=[self.params[0][0], self.params[0][1], chance, 0.001],
                                           param_free=self.params[1])  # parameters as [scale, slope, gamma, lambda]
            fit_dict[key]['data'] = data_matrix
            fit_dict[key]['metrics'] = metrics
            fit_dict[key]['options'] = options

        return fit_dict

    def _to_df(self):
        fit_dict = self.fit()
        df = pd.DataFrame(columns=['hue_angle', 'ntrial', 'JND', 'JND_SD', 'PSE', 'PSE_SD'])
        for key, fit in fit_dict.items():
            if len(key) > 1:
                key = key[1]
            df.loc[key, 'hue_angle'] = fit['hue_angle']
            df.loc[key, 'ntrial'] = fit['ntrial']
            df.loc[key, 'PSE'] = fit['metrics']['MAP'][0]
            df.loc[key, 'JND'] = fit['metrics']['MAP'][1]
            df.loc[key, 'PSE_SD'] = fit['metrics']['SD'][0]
            df.loc[key, 'JND_SD'] = fit['metrics']['SD'][1]
        df = df.reset_index()
        return df

    def plot_pf_curve(self):
        """
        Plot PFs from nonlinear least squares fitting results.

        """
        fit_dict = self.fit()
        num = len(fit_dict)

        if self.ylabel == 'response':
            ylabel = 'Prob. Response as larger hue angles'
            n_col = int(num / 2)
            xlim = [-18, 18]
            ylim = [-0.05, 1.05]
        else:
            ylabel = 'Prob. Correct judge'
            n_col = int(num / 4)
            xlim = [0, 18]
            ylim = [0.45, 1.05]

        if num == 1:
            fig, axes = plt.subplots(num, 1, figsize=plot_config['figsize'])
        else:
            fig, axes = plt.subplots(2, n_col, figsize=plot_config['figsize'])

        for idx, key in enumerate(fit_dict):

            this_fit = fit_dict[key]
            hue_angle = this_fit['hue_angle']
            ntrial = this_fit['ntrial']
            data = this_fit['data']
            options = this_fit['options']
            # Determine which values to use for vector of parameters
            param_guess = np.zeros(4)
            counter = 0
            for keys in options['param_free']:
                if keys is True:
                    param_guess[counter] = this_fit['metrics']['MAP'][counter]
                elif keys is False:
                    param_guess[counter] = options['param_ests'][counter]
                counter += 1

            if num == 1:
                ax = axes

            if this_fit['ylabel'] == 'response':
                ax = axes.flatten()[idx]
                label = None
                color = color4plot(hue_angle)[0]

            elif this_fit['ylabel'] == 'correct':
                ax = axes.flatten()[int(np.floor(idx / 2))]
                if key[1] < 0:
                    label = 'minus'
                    color = 'coral'
                else:
                    label = 'plus'
                    color = 'skyblue'

            # color_code = color4plot(hue_angle)[0]

            # hue_angles = np.array([float(k) for k in fit_dat.keys()])
            # color = colorcodes[int((key - self.first_angle) / self.first_angle / 2)]
            for i in range(data[:, 0].shape[0]):
                ax.scatter(data[i, 0],
                           data[i, 1] / data[i, 2],
                           color=color,
                           s=data[i, 2] * 2,
                           alpha=0.5,
                           zorder=5,
                           marker='o')

            # Generate smooth curve from fitted function
            # x_max = data[:, 0].max()
            # x_min = data[:, 0].min()
            x_min = xlim[0]
            x_max = xlim[1]
            x_est = np.linspace(x_min, x_max, 50)

            y_pred = _psyfunction(x_est,
                                  param_guess[0],
                                  param_guess[1],
                                  param_guess[2],
                                  param_guess[3],
                                  options['sigmoid_type'])
            ax.plot(x_est, y_pred, '-', color=color, label=label)  # plot fitted curve

            # ax.axhline(y=thre_y, color='grey', linestyle='dashed', linewidth=1, zorder=1, alpha=0.5)
            # ax.axvline(x=this_fit['metrics']['threshold'], color='grey', linestyle='dashed', linewidth=1, zorder=1, alpha=0.5)

            ax.plot([x_min, this_fit['metrics']['threshold']],
                    [options['threshold'], options['threshold']],
                    color='grey',
                    linestyle='dotted',
                    linewidth=1,
                    zorder=1,
                    alpha=0.8)
            ax.plot([this_fit['metrics']['threshold'], this_fit['metrics']['threshold']],
                    [0, options['threshold']],
                    color='grey',
                    linestyle='dotted',
                    linewidth=1,
                    zorder=1,
                    alpha=0.8)

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
        plt.setp(y_ax, ylabel=ylabel)

        plt.setp(ax.get_xticklabels(), fontsize=plot_config['tick_size'])
        plt.setp(ax.get_yticklabels(), fontsize=plot_config['tick_size'])

        # fig.suptitle(self.sub[0:2] + '_' + str(ntrial) + 'trials', fontsize=plot_config['title_fontsize'])
        fig.suptitle(str(ntrial) + ' trials', fontsize=plot_config['title_fontsize'])
        plt.legend()
        plt.show()


# s05_ll = LoadData('s05', data_path='data', sel_par=['LL_2x2']).read_data()
# FitPf = FitPf_bayes(s05_ll, ylabel='correct', func='norm', params=[[None, None], [True, True, False, False]])
# fit_dict = FitPf.fit()
# fit_df = FitPf._to_df()
# FitPf.plot_pf_curve()