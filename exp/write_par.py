# !/usr/bin/env python3.7
# -*- coding: utf-8 -*-
""" 
Created on 27.07.21

@author yannan su
"""
import yaml
import numpy as np


def write_par_set(file_path, stim, noise_condition, noise_level=0.):
    """
    Write parameters to a YAML file.

    :param file_path:   par-file path
    :param stim:        example:   stim = {'hue_1': 22.5, ' hue_5': 202.5}
    :param seed:        seed for random permutation of the stimuli
    :param hue_num:     hue numbers
    :param theta:       customize a theta value list
    :param min_max:     [min value, max value] for simple and quest methods
    :param start_val:   start value for simple and quest methods
    :param std:         stimulus noise level (a single number or a list of numbers)
    :param step_type:   'db', 'lin', 'log' (None if method is not 'simple')
    :param up_down:     tuple with the up-down rule settings (up, down)
    :param p_threshold: targeting probability threshold for the quest method

    Example:
    write_par('config/cn2x2x8_crs_test.yaml', noise='cross', method='quest', hue_num=8, std=5.06)

    """
    idx = 0
    par_dict = {}
    for key, val in stim.items():
        for ii in range(2):
            stim_num = 'stimulus_' + str(idx)
            idx += 1
            par_dict[stim_num] = {}
            if idx % 2 == 0:
                par_dict[stim_num]['label'] = key + 'm'
                par_dict[stim_num]['stairDirection'] = -1.0
            else:
                par_dict[stim_num]['label'] = key + 'p'
                par_dict[stim_num]['stairDirection'] = 1.0

            par_dict[stim_num]['noise_condition'] = noise_condition
            par_dict[stim_num]['standard'] = val
            par_dict[stim_num]['width'] = noise_level
            par_dict[stim_num]['hue_range'] = [val - 30., val + 30.]
            par_dict[stim_num]['hue_num'] = 45
            par_dict[stim_num]['stairType'] = 'simple'
            par_dict[stim_num]['stepType'] = 'lin'
            par_dict[stim_num]['startVal'] = 5.
            par_dict[stim_num]['minVal'] = .5
            par_dict[stim_num]['maxVal'] = 20.
            par_dict[stim_num]['nUp'] = 1
            par_dict[stim_num]['nDown'] = 2
            par_dict[stim_num]['stepSizes'] = [2.0, 1.0, 0.5]
            par_dict[stim_num]['nReversals'] = 2
    with open(file_path, 'w') as file:
        yaml.dump(par_dict, file, default_flow_style=False, sort_keys=False)


# write_par_set('config/par_a/LL_2x2_set5.yaml', {'hue_9': 0., 'hue_13': 180.}, 'L-L', noise_level=0.)
# write_par_set('config/par_a/LL_2x2_set6.yaml', {'hue_11': 90., 'hue_15': 270.}, 'L-L', noise_level=0.)
# write_par_set('config/par_a/LL_2x2_set7.yaml', {'hue_10': 45., 'hue_14': 225.}, 'L-L', noise_level=0.)
# write_par_set('config/par_a/LL_2x2_set8.yaml', {'hue_12': 135., 'hue_16': 315.}, 'L-L', noise_level=0.)


