#!/user/bin/env python
# coding=utf-8
"""
Read YAML file and convert it to a python dictionary.

@author: yannansu
@created at: 19.03.21 21:53
"""
import yaml


def yml2dict(file_path):
    """
    Load and read a YAML file.

    :param file_path:   YAML file path
    :return:            a dictionary converted from YAML
    """
    with open(file_path) as file:
        par_dict = yaml.load(file, Loader=yaml.FullLoader)
    return par_dict


