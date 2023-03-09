#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
at: 17.05.22 11:43

Experiment for determining individual's four unique hues.
Each run repeat the measurements #n_rep times, e,g, n_rep = 10

Quick start from bash:
python3 unique_hue.py --s sub --c config/exp_config.yaml --p config/unique_hues.yaml --r data/unique_hue

"""

import numpy as np
import os
import time
from bisect import bisect_left
from psychopy import monitors, visual, core, event, data, misc
import argparse
from pyiris.colorspace import ColorSpace
from write_data import WriteData, WriteActivity
from yml2dict import yml2dict


class Exp:
    def __init__(self, subject, cfg_file, par_file, res_dir):
        """
        :param subject:     subject, e.g. s00 - make sure you have colorspace files for s00
        :param cfg_file:    config file path, can share the configs of the main experiment 'config/exp_configs.yaml'
        :param par_file:    parameter file path, 'config/unique_hues.yaml'
        :param res_dir:     results directory, 'data/unique_hue'
        """
        self.subject = subject
        if not res_dir:
            res_dir = 'data/'
        self.res_dir = res_dir
        self.cfg_file = cfg_file
        self.par_file = par_file

        self.cfg = yml2dict(self.cfg_file)
        self.par = yml2dict(self.par_file)

        mon_settings = yml2dict(self.cfg['monitor_settings_file'])
        self.monitor = monitors.Monitor(name='ViewPixx Lite 2000A', distance=mon_settings['preferred_mode']['distance'])

        # Only need to run and save the monitor setting once!!!
        # The saved setting json is stored in ~/.psychopy3/monitors and should have a backup in config/resources/
        self.monitor = monitors.Monitor(name=mon_settings['name'],
                                        distance=mon_settings['preferred_mode']['distance'])
        self.monitor.setSizePix((mon_settings['preferred_mode']['width'],
                                 mon_settings['preferred_mode']['height']))
        self.monitor.setWidth(mon_settings['size']['width'] / 10.)
        self.monitor.saveMon()

        subject_path = self.cfg['subject_isolum_directory'] + '/' + 'colorspace_' + subject + '.json'
        self.bit_depth = self.cfg['depthBits']

        self.colorspace = ColorSpace(bit_depth=self.bit_depth,
                                     chromaticity=self.cfg['chromaticity'],
                                     calibration_path=self.cfg['calibration_file'],
                                     subject_path=subject_path)

        # self.gray = self.colorspace.lms_center  # or use lms center
        self.gray = np.array([self.colorspace.gray_level, self.colorspace.gray_level, self.colorspace.gray_level])
        self.colorspace.create_color_list(hue_res=0.05,
                                          gray_level=self.colorspace.gray_level)  # Make sure the reslution is fine enough - 0.05 should be good
        self.colorlist = self.colorspace.color_list[0.05]
        self.gray_pp = self.colorspace.color2pp(self.gray)[0]
        self.win = visual.window.Window(size=[mon_settings['preferred_mode']['width'],
                                              mon_settings['preferred_mode']['height']],
                                        monitor=self.monitor,
                                        units=self.cfg['window_units'],
                                        fullscr=True,
                                        colorSpace='rgb',
                                        color=self.gray_pp,
                                        mouseVisible=False)

        self.idx = time.strftime("%Y%m%dT%H%M", time.localtime())  # index as current date and time

        self.text_height = self.cfg['text_height']

    def run_exp(self):
        """
        Main function for starting experiments.
        :return:
        """

        unique_hues = list(self.par.keys())
        start_val_range = 20
        start_val_num = 40

        n_rep = 5
        n_trial = len(unique_hues) * n_rep

        texts = self.make_texts()

        # welcome
        texts['welcome'].draw()
        self.win.flip()
        event.waitKeys()

        # init data files
        InitActivity = WriteActivity(self.subject, self.idx, dir_path=self.res_dir)
        idx = time.strftime("%Y%m%dT%H%M", time.localtime())  # index as current date and time
        InitData = WriteData(self.subject, idx, self.res_dir)
        data_file = InitData.head(self.cfg_file, self.par_file)

        trial_count = 0

        # iterate trials
        for i_rep in np.arange(n_rep):
            np.random.shuffle(unique_hues)
            for hue in unique_hues:
                trial_count += 1
                start_vals = np.linspace(self.par[hue][0]['guess_val'] - start_val_range,
                                         self.par[hue][0]['guess_val'] + start_val_range,
                                         num=start_val_num,
                                         endpoint=True)
                theta = np.random.choice(start_vals, 1)

                dat = {'sub': self.subject,
                       'i_rep': float(i_rep),
                       'unique_hue': hue,
                       'random_start': float(theta),
                       'estimate': None,
                       'RT': None
                       }

                text = visual.TextStim(self.win,
                                       text=self.par[hue][1]['text'],
                                       pos=(0, 10),
                                       color='black',
                                       height=self.text_height)

                mouse = event.Mouse(win=self.win, visible=False)
                mouse.setPos(0.)
                _, y_pos = mouse.getPos()

                mouse_lim = 4  # fine movement can return 1 deg resolution with this parameter
                finish_current_trial = False
                reactClock = core.Clock()

                while finish_current_trial is False:
                    color_patch = self.make_color_patch(theta)
                    color_patch.draw()
                    text.draw()
                    self.win.flip()

                    # change theta by moving mouse cursor vertically
                    _, y = mouse.getPos()  # update y-position
                    # change theta based on position change - the range depends on mouse_lim
                    d_theta = mouse_lim * (y - y_pos)
                    # Update pos and theta
                    y_pos = y
                    theta += d_theta

                    # print(theta)

                    if event.getKeys('space', timeStamped=reactClock):
                        finish_current_trial = True
                        test_theta, _ = self.closest_hue(theta)
                        dat['estimate'] = float(test_theta)
                        dat['RT'] = float(np.round(reactClock.getTime(), 2))

                        InitData.write(trial_count, 'trial', dat)

                    if event.getKeys('escape', timeStamped=reactClock):
                        texts['confirm_escape'].draw()
                        self.win.flip()
                        for key_press in event.waitKeys():
                            if key_press == 'y':
                                # Save escape info
                                InitActivity.write(self.cfg_file, self.par_file, data_file, status='escape')
                                core.quit()

                mask = self.make_checkerboard_mask()
                mask.draw()
                self.win.flip()
                core.wait(0.5)
        InitActivity.write(self.cfg_file, self.par_file, data_file, status='completed')

        texts['goodbye'].draw()
        self.win.flip()
        event.waitKeys()

    def make_color_patch(self, theta):
        """
        Create a color patch for adjustment.

        :param theta:       hue angle
        :return:            a colorpatch as a psychopy Rect stimulus
        """

        closest_theta, closest_rgb = self.closest_hue(theta)
        color_patch = visual.Rect(win=self.win,
                                  width=10.,
                                  height=10.,
                                  pos=(0, 0),
                                  lineWidth=0,
                                  fillColor=closest_rgb)
        return color_patch

    def make_checkerboard_mask(self):
        """
        Create a color checkerboard mask.
        :return:
        """
        horiz_n = 35
        vertic_n = 25
        rect = visual.ElementArrayStim(self.win,
                                       units='norm',
                                       nElements=horiz_n * vertic_n,
                                       elementMask=None,
                                       elementTex=None,
                                       sizes=(2 / horiz_n, 2 / vertic_n))
        rect.xys = [(x, y)
                    for x in np.linspace(-1, 1, horiz_n, endpoint=False) + 1 / horiz_n
                    for y in np.linspace(-1, 1, vertic_n, endpoint=False) + 1 / vertic_n]

        rect.colors = [self.closest_hue(theta=x)[1]
                       for x in
                       np.random.randint(0, high=360, size=horiz_n * vertic_n)]

        return rect

    def closest_hue(self, theta):
        """
        Tool function:
        Given a desired hue angle, to find the closest hue angle and the corresponding rgb value.

        :param theta:   desired hue angle (in degree)
        :return:        closest hue angle, closest rgb values
        """

        hue_angles = np.array(self.colorlist['hue_angles'])
        if theta < 0:
            theta += 360
        if theta >= 360:
            theta -= 360
        closest_theta, pos = np.array(self.take_closest(hue_angles, theta))
        closest_rgb = self.colorlist['rgb'][pos.astype(int)]
        closest_rgb = self.colorspace.color2pp(closest_rgb)[0]
        return np.round(closest_theta, 2), closest_rgb

    def take_closest(self, arr, val):
        """
        Tool function:
        Assumes arr is sorted. Returns closest value to val (could be itself).
        If two numbers are equally close, return the smallest number.

        :param arr:   sorted array
        :param val:   desired value
        :return:      [closest_val, closest_idx]
        """
        pos = bisect_left(arr, val)
        if pos == 0:
            return [arr[0], pos]
        if pos == len(arr):
            return [arr[-1], pos - 1]
        before = arr[pos - 1]
        after = arr[pos]
        if after - val < val - before:
            return [after, pos]
        else:
            return [before, pos - 1]

    def make_texts(self):
        """
        Create texts.
        :return:
        """
        texts = {}
        # Define welcome message
        texts["welcome"] = visual.TextStim(self.win,
                                           text=f"Welcome! \n \n"
                                                f"Please follow the instruction to adjust the color by mouse, "
                                                f"until a unique color appears. \n"
                                                f"Then press space key to confirm. \n"
                                                f"Ready? \n"
                                                f"Press any key to start this session :)",
                                           pos=(0, 0),
                                           color='black',
                                           height=self.text_height)

        # Define goodbye message
        texts["goodbye"] = visual.TextStim(self.win,
                                           text=f"Well done! \n \n"
                                                f"You have completed this session. \n"
                                                f"Press any key to quit:)",
                                           pos=(0, 0),
                                           color='black',
                                           height=self.text_height)

        # Define escape confirm
        texts["confirm_escape"] = visual.TextStim(self.win,
                                                  text=f"Are you sure to quit (Y/N)? \n",
                                                  pos=(0, 0),
                                                  color='black',
                                                  height=self.text_height)

        texts['trial_count'] = visual.TextStim(self.win,
                                               pos=(0, -10),
                                               color='black',
                                               height=self.text_height * .5)

        return texts


# """
# Run from bash
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='Subject name')
    parser.add_argument('-c', help='Configuration file')
    parser.add_argument('-p', help='Parameter file')
    parser.add_argument('-r', help='Results folder')

    args = parser.parse_args()

    Exp(subject=args.s, cfg_file=args.c, par_file=args.p, res_dir=args.r).run_exp()
# """

"""
# Test
Exp(subject='s01',
    cfg_file='config/exp_config.yaml',
    par_file='config/unique_hues.yaml',
    res_dir='data/unique_hue').run_exp()
"""
