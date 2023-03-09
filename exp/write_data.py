#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

# python version 3.7.6
""" 
@author: yannansu
@created at: 19.03.21 10:38

This module contains all writing and reading functions for configuration files for the color-noise experiment.

Functions for writing:
    function write_cfg:     write experiment config file *.yaml
    function write_par:     write parameter file *.yaml
    class WriteXpp:         write session log file *.yaml
    function write_xrl:     write subject log file *.xrl

Functions for reading:
    function read_yml:      read all types of *.yaml
    function read_xrl:      read subject log file *.xrl

"""
import os
import numpy as np
import yaml
import sys
import xlsxwriter
import copy
import psychopy
# from .yml2dict import yml2dict
from exp.yml2dict import yml2dict


class WriteData:
    def __init__(self, subject, idx, dir_path='data'):
        """
        Create a log YAML file for one session.

        :param subject:     subject name [string]
        :param idx:         date and time [string]
        :param dir_path:    the directory for storing. default: data/subject
        """
        base = os.path.join(dir_path, subject, 'raw_data')
        if not os.path.exists(base):
            os.makedirs(base)

        self.idx = idx
        self.file_path = os.path.join(base, subject + '_' + idx + '.yaml')
        self.f = open(self.file_path, 'w+')

    def head(self, cfg_file, par_file):
        """
        Copy the metadata from experiment config and parameter file into the head part of this log file.

        :param cfg_file:    experiment config file path
        :param par_file:    parameter file path
        :return:            the log file path
        """
        info = {'time': self.idx,
                'cfg_file': cfg_file,
                'par_file': par_file}
        # yaml.safe_dump(info, self.f, default_flow_style=False, sort_keys=False)

        cfg_dict = yml2dict(cfg_file)
        par_dict = yml2dict(par_file)
        yaml.safe_dump({**info, **cfg_dict, **par_dict}, self.f, default_flow_style=False, sort_keys=False)
        return self.file_path

    def write(self, count, trial_type, result_dict):
        """
        Append log of every single trials in iterations to this log file.

        cond, patch_xlims, rot, disp_intensity, press_key, judge, react_time, trial_stamp

        :param count:           count of this trial
        :param trial_type:      trial type, 'trial' or 'warmup'
        :param result_dict:     a dictionary of results
        """
        if trial_type != 'trial' and trial_type != 'warmup':
            raise ValueError("The trial type is incorrect!")
        this_trial = trial_type + '_' + str(count)
        trial_dict = {this_trial: result_dict}
        yaml.dump(trial_dict, self.f, default_flow_style=False, sort_keys=False)
        self.f.flush()


class WriteActivity:
    def __init__(self, subject, idx, dir_path='data'):
        """
        Create a log YAML file for one session.

        :param subject:     subject name [string]
        :param idx:         date and time [string]
        :param dir_path:    the directory for storing. default: data/subject
        """
        base = os.path.join(dir_path, subject)
        if not os.path.exists(base):
            os.makedirs(base)

        self.file_path = os.path.join(base, subject + '.yaml')

        if os.path.exists(self.file_path):
            self.f = open(self.file_path, 'a+')
        else:
            self.f = open(self.file_path, 'w+')
        self.idx = idx

    def write(self, cfg_file, par_file, data_file, status=None):
        """
        :param cfg_file:    experiment config file path
        :param par_file:    parameter file path
        :param data_file:   parameter file path
        :param status:
        :return:            the log file path
        """
        if yml2dict(self.file_path) is None:
            session_count = 1
        else:
            session_count = len(yml2dict(self.file_path).keys()) + 1
        start_info = {'session_count': session_count,
                      'session_idx': self.idx,
                      'cfg_file': cfg_file,
                      'par_file': par_file,
                      'data_file': data_file,
                      'status': status}
        session_dict = {self.idx: start_info}
        yaml.safe_dump(session_dict, self.f, default_flow_style=False, sort_keys=False)
        # self.f.flush()
        return self.file_path

    # def stop(self, status):
    #     """
    #
    #     :param status: 'completed', 'practice', 'userbreak'
    #     :return:
    #     """
    #     this_session = yml2dict(self.file_path)[self.idx]
    #     this_session['status'] = status
    #     session_dict = {self.idx: this_session}
    #     yaml.safe_dump(session_dict, self.f, default_flow_style=False, sort_keys=False)
    #     self.f.flush()
