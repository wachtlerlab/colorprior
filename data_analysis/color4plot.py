#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 23.03.21 11:40
"""
import pyiris.colorspace as pyc
import numpy as np
import matplotlib.pyplot as plt


def color4plot(theta_array):
    """
    Generate colors for plotting.

    """
    colorspace = pyc.ColorSpace(bit_depth=8,
                                calibration_path='config/resources/10bit/calibration_10bit.json',
                                chromaticity=0.2,
                                gray_level=0.8)
    # angles = np.linspace(0, 360, num=num, endpoint=False) + 22.5
    rgbs = colorspace.dklc2rgb(phi=theta_array * np.pi / 180)
    return rgbs


def plot_color_circle():
    cp = pyc.ColorSpace(bit_depth=8,
                        chromaticity=0.2,
                        calibration_path='config/resources/10bit/calibration_20210406.json',
                        gray_level=0.8)
    deg_array = np.linspace(0, 360, 360)
    rad_array = np.linspace(0, 2 * np.pi, 360)
    rgbs = cp.dklc2rgb(phi=deg_array * np.pi / 180)

    x = np.sin(rad_array + np.pi / 2)
    y = np.cos(rad_array - np.pi / 2)

    sel_array = np.linspace(22.5/180 * np.pi, (360+22.5)/180 * np.pi, 8, endpoint=False)
    x_sel = np.sin(sel_array + np.pi / 2)
    y_sel = np.cos(sel_array - np.pi / 2)

    fig, ax = plt.subplots(figsize=(3.45, 3.45))
    ax.scatter(x * 0.5, y * 0.5, c=rgbs, s=250)
    ax.scatter(x_sel * 0.5, y_sel * 0.5, s=100, marker='o', c='none', edgecolors='gray', lw=2)

    plt.axis('off')
    # plt.savefig('data_analysis/figures/color_circle_no_marker.pdf')
    plt.savefig('data_analysis/figures/color_circle.pdf')
    plt.show()

# plot_color_circle()
