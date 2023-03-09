#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 19.03.21 10:38

Example usage:
LD = LoadData('test', sel_par=['set3'])
activ_df = LD.read_activity()
df = LD.read_data('test_LL.csv')
"""
import numpy as np
import pandas as pd
import datetime
import os
import yaml
from .yml2dict import yml2dict


class LoadData:
    def __init__(self, sub, data_path=None,
                 sel_cfg=None, sel_par=None,
                 sel_ses=None, rm_ses=None, start_date=None,
                 sel_ses_idx=None,
                 include_practice=False):
        """

        :param sub:
        :param data_path:
        :param sel_cfg:
        :param sel_par:
        :param sel_ses:
        :param rm_ses:
        :param start_date:
        :param sel_ses_idx:
        :param include_practice:
        """
        self.sub = sub
        self.data_path = data_path
        if self.data_path is None:
            self.data_path = 'data'
        self.sel_cfg = sel_cfg
        self.sel_par = sel_par
        self.sel_ses = sel_ses
        self.rm_ses = rm_ses
        self.start_date = start_date
        self.sel_ses_idx = sel_ses_idx
        self.include_practice = include_practice

    def read_activity(self):
        """
        Read subject activity log and select sessions.

        :return:  a dataframe listing a summary of the selected session
        """
        # Read xrl file line by line
        activ_log = yml2dict(os.path.join(self.data_path, self.sub, self.sub + '.yaml'))
        activ_df = pd.DataFrame(activ_log).T

        # Select finished sessions
        if self.include_practice:
            activ_df = activ_df[activ_df.status.str.contains('practice|completed')]
        else:
            activ_df = activ_df[activ_df.status == 'completed']

        # Filter data by input selector
        if self.sel_cfg is not None:
            cfg_pattern = '|'.join(self.sel_cfg)
            activ_df = activ_df[activ_df.cfg_file.str.contains(cfg_pattern)]
        if self.sel_par is not None:
            par_pattern = '|'.join(self.sel_par)
            activ_df = activ_df[activ_df.par_file.str.contains(par_pattern)]

        # Restrict specific sessions by date or time
        if self.sel_ses is not None:
            activ_df = activ_df.filter(like=self.sel_ses, axis=0)
        if self.rm_ses is not None:
            activ_df = activ_df.drop(self.rm_ses, axis=0)
        if self.start_date is not None:
            activ_df = activ_df[activ_df.index >= self.start_date]
        if self.sel_ses_idx is not None:
            activ_df = activ_df.groupby('par_file').nth(self.sel_ses_idx)

        activ_df = activ_df.reset_index()
        if 'index' in activ_df.columns:
            activ_df = activ_df.drop(columns=['index'])

        return activ_df

    def read_data(self, save_csv=None, convert_judge=True):
        """
        Read data from selected session.

        :param save_csv:            '*.csv', saved csv file name in 'data/subject_xx/'
        :param convert_judge:       convert correctness to 'resp_as_larger' if True;
                                    False if you are fitting to correctness
        :return:                    a dataframe of trial-based data of all selected sessions
        """
        # Read activity log
        activ_df = self.read_activity()

        # Read data files
        yml_list = activ_df.data_file.to_list()
        df_list = []

        for session_idx, yml in zip(activ_df.session_idx, yml_list):
            yml_data = pd.DataFrame({k: v for k, v in yml2dict(yml).items()
                                     if (k.startswith('trial') and any(c.isdigit() for c in k))
                                     }).T
            yml_data['time_index'] = session_idx
            yml_data.reset_index()

            if convert_judge is True:
                yml_data['resp_as_larger'] = yml_data['judge']
                idx = yml_data['actual_intensity'] < 0
                yml_data.loc[idx, 'resp_as_larger'] = 1 - yml_data.loc[idx, 'judge']

            df_list.append(yml_data)
        df = pd.concat(df_list)
        if save_csv is not None:
            df.to_csv(os.path.join(self.data_path, self.sub, save_csv))
        return df



