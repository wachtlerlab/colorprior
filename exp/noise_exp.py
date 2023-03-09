#!/user/bin/env python
# coding=utf-8
"""
@author: yannansu
@created at: 06.04.21 10:11

A module for running an additional experiment to determine a subject's noise level

Quick testing the experiment from bash:
python main_exp.py --s test --c config/test_cfg.yaml --p  --feedback False

"""

import numpy as np
import os
import time
from bisect import bisect_left
from psychopy import monitors, visual, core, event, data, misc
import argparse
from pyiris.colorspace import ColorSpace
import write_data
from yml2dict import yml2dict


class Exp:
    def __init__(self, subject, cfg_file, par_file, res_dir, feedback):
        self.subject = subject
        if not res_dir:
            res_dir = 'data/noise_test/'
        self.res_dir = res_dir
        self.cfg_file = cfg_file
        self.par_file = par_file
        self.feedback = feedback

        self.cfg = yml2dict(self.cfg_file)
        self.param = yml2dict(self.par_file)
        self.conditions = [dict({'stimulus': key}, **value)
                           for key, value in self.param.items()
                           if key.startswith('stimulus')]

        mon_settings = yml2dict(self.cfg['monitor_settings_file'])
        self.monitor = monitors.Monitor(name='ViewPixx Lite 2000A', distance=mon_settings['preferred_mode']['distance'])

        """
        # Only need to run and save the monitor setting once!!!
        # The saved setting json is stored in ~/.psychopy3/monitors and should have a backup in config/resources/
        self.monitor = monitors.Monitor(name=mon_settings['name'],
                                        distance=mon_settings['preferred_mode']['distance'])
        self.monitor.setSizePix((mon_settings['preferred_mode']['width'],
                                 mon_settings['preferred_mode']['height']))
        self.monitor.setWidth(mon_settings['settings']['left-right-size'])
        self.monitor.saveMon()
        """

        subject_path = self.cfg['subject_isolum_directory'] + '/colorspace_' + subject + '.json'
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
        self.trial_dur = self.cfg['trial_dur']
        self.trial_nmb = self.cfg['trial_nmb']
        self.warmup_nmb = self.cfg['warmup_nmb']
        self.total_nmb = (self.trial_nmb + self.warmup_nmb) * len(self.conditions)

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

    def get_disp_val(self, cond, rot):
        """
        Tool function:
        Check whether the stimuli are truly displayed in the given monitor resolution,
        and if not, the rotation given by a staircase should be corrected by realizable values.

        :param cond:   a condition of a staircase [dictionary]
        :param rot:    rotation angle [degree] relative to cond['standard'] given by a staircase
        :return:       closest rotation angle [degree]
        """
        disp_standard, _ = self.closest_hue(cond['standard'])  # actually displayed standard
        stair_test = cond['standard'] + rot  # calculated test value
        if stair_test < 0:
            stair_test += 360
        disp_test, _ = self.closest_hue(stair_test)  # actually displayed test value
        disp_intensity = disp_test - disp_standard  # actually displayed intensity (i.e. difference)
        if disp_intensity > 300:
            disp_intensity = (disp_test + disp_standard) - 360
        return disp_intensity

    def create_bar(self, hue_range, hue_num):
        """
        Create a color bar stimulus covering a given hue angle range.

        :param hue_range:   [min_hue_angle, max_hue_angle] in degree
        :param hue_num:     the number of shown hues (default as 45; should be many to make the hues appear continuous)
        :return:            a colorbar as a psychopy ImageStim stimulus
        """
        hue_calculated = np.linspace(hue_range[0], hue_range[1], hue_num)
        closest_theta, closest_rgb = zip(*[self.closest_hue(v) for v in hue_calculated])
        pos = [[x, self.cfg['bar_ypos']] for x
               in np.linspace(self.cfg['bar_xlim'][0], self.cfg['bar_xlim'][1], hue_num)]
        colorbar = visual.ElementArrayStim(win=self.win,
                                           fieldSize=self.cfg['bar_size'],
                                           xys=pos,
                                           nElements=hue_num,
                                           elementMask=None,
                                           elementTex=None,
                                           sizes=self.cfg['bar_size'][1])
        colorbar.colors = np.array(closest_rgb)
        return colorbar

    def patch_stim(self, xlim, ylim):
        """
        Set properties for standard and test stimuli.

        :param xlim:    x-axis limit [x_1, x_2]
        :param ylim:    y-axis limit [y_1, y_2]
        :return:        an array of circular patches as a Psychopy ElementArrayStim stimulus
        """
        n = int(np.sqrt(self.cfg['patch_nmb']))
        pos = [[x, y]
               for x in np.linspace(xlim[0], xlim[1], n)
               for y in np.linspace(ylim[0], ylim[1], n)]
        patch = visual.ElementArrayStim(win=self.win,
                                        fieldSize=self.cfg['field_size'],
                                        xys=pos,
                                        nElements=self.cfg['patch_nmb'],
                                        elementMask='circle',
                                        elementTex=None,
                                        sizes=self.cfg['patch_size'])
        return patch

    def rand_color(self, theta, width, npatch):
        """
        Generate the hues of a stimulus array with noise.

        :param theta:   the mean hue angle of all hues in the stimulus array
        :param width:   the half width of a uniform distribution in the stimulus array
        :param npatch:  the number of patches in the stimulus array
        :return:
            - angle:    an array of npatch hue angles
            - rgb:      an array of npatch rgb values
        """
        # Sample from uniform distribution with equivalent spaces
        noise = np.linspace(theta - width, theta + width, int(npatch), endpoint=True)
        # Shuffle the position of the array
        np.random.shuffle(noise)
        angle, rgb = zip(*[self.closest_hue(theta=n) for n in noise])
        return angle, np.array(rgb)

    def choose_con(self, noise, standard, test, width):
        """
        Generate stimuli color rgb values with the chosen noise condition.

        :param noise:       noise condition as 'L-L', 'H-H', 'L-H', or 'H-L'
        :param standard:    the (average) hue angle of the standard stimulus
        :param test:        the (average) hue angle of the test stimulus
        :param width:       the half width of a uniform distribution in the stimulus array
        :return:
            - sColor:       an array of rgb values of the standard stimulus (shape as npatch x 1)
            - tColor:       an array of rgb values of the test stimulus     (shape as npatch x 1)
        """
        sColor = None
        tColor = None

        if noise == 'L-L':  # low - low noise
            _, sColor = self.closest_hue(theta=standard)
            _, tColor = self.closest_hue(theta=test)

        elif noise == 'L-H':  # low noise in standard, high noise in test
            _, sColor = self.closest_hue(theta=standard)
            _, tColor = self.rand_color(test, width, self.cfg['patch_nmb'])

        # elif noise == 'H-L':  # high noise in standard, low noise in test
        #     _, sColor = self.rand_color(test, width, self.cfg['patch_nmb'])
        #     _, tColor = self.closest_hue(theta=test)

        elif noise == 'H-H':  # high - high noise
            _, sColor = self.rand_color(standard, width, self.cfg['patch_nmb'])
            _, tColor = self.rand_color(test, width, self.cfg['patch_nmb'])

        else:
            print("No noise condition corresponds to the input!")

        return sColor, tColor

    def run_trial(self, width, cond, patch_xlims, count, InitActivity, data_file):
        """
        Run a single trial.

        :param width:       the half width of a uniform distribution in the stimulus array
        :param cond:        stimulus and staircase condition of the this trial [dictionary]
        :param patch_xlims: xlim for defining standard and test patch positions [2x2 array]
        :param count:       the count of trial
        :return:
            - judge:            subject's response as 0 or 1
            - react_time:       reaction time in sec
            - trial_time_start: the time stamp when a trial starts
        """
        ref = self.create_bar(cond['hue_range'], cond['hue_num'])

        sPatch_xlim, tPatch_xlim = patch_xlims

        sPatch = self.patch_stim(sPatch_xlim, self.cfg['standard_ylim'])
        tPatch = self.patch_stim(tPatch_xlim, self.cfg['test_ylim'])

        # Set colors of two stimuli
        standard = cond['standard']  # standard should be fixed
        test = cond['test']
        rot = test - standard
        sPatch.colors, tPatch.colors = self.choose_con('H-H',
                                                       standard,
                                                       test,
                                                       width)

        # Fixation cross & Number of trial
        fix = visual.TextStim(self.win,
                              text="+",
                              pos=[0, 0],
                              height=0.6,
                              color='black')
        num = visual.TextStim(self.win,
                              text=f"trial {count} of {self.total_nmb} trials",
                              pos=[0, -10],
                              height=0.5,
                              color='black')

        trial_time_start = time.time()

        # Present the standard and the test stimuli together with the reference
        fix.draw()
        num.draw()
        ref.draw()
        self.win.flip()
        core.wait(0.5)

        fix.draw()
        num.draw()
        ref.draw()
        sPatch.draw()
        tPatch.draw()
        self.win.flip()

        key_press = None
        judge = None
        react_time_stop = -1
        react_time_start = time.time()

        core.wait(self.trial_dur)

        # Check whether the how the hue angle change from left stimulus to right stimulus:
        # - if increase, change_direction = up
        # - if decrease, change_direction = down
        # Notice that the corresponding changing direction of the reference bar is always "up", i.e. increasing.
        if (sum(sPatch_xlim) < 0 < rot) or (sum(sPatch_xlim) > 0 > rot):
            change_direction = 'up'
        else:
            change_direction = 'down'

        # Refresh and show a colored checkerboard mask for 0.5 sec
        mask_dur = 0.5
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
        rect.draw()
        self.win.flip()
        core.wait(mask_dur)

        # If response is given during the mask
        if judge is None:
            get_keys = event.getKeys(['up', 'down', 'escape'])
            key_press = get_keys
            if change_direction in get_keys:
                judge = 1  # correct
                react_time_stop = time.time()
            elif ('up' in get_keys and change_direction == 'down') or \
                    ('down' in get_keys and change_direction == 'up'):
                judge = 0  # incorrect
                react_time_stop = time.time()
            elif 'escape' in get_keys:
                InitActivity.write(self.cfg_file, self.par_file, self.res_dir, data_file, status='escape')
                core.quit()

        # Refresh and wait for response (if no response was given in the pause mode or during mask)
        self.win.flip()
        fix.draw()
        num.draw()
        self.win.flip()

        # If no response in the pause mode or during the mask
        if judge is None:
            for wait_keys in event.waitKeys(keyList=['up', 'down', 'escape']):
                key_press = wait_keys
                if wait_keys == change_direction:
                    judge = 1  # correct
                    react_time_stop = time.time()
                elif (wait_keys == 'up' and change_direction == 'down') or \
                        (wait_keys == 'down' and change_direction == 'up'):
                    judge = 0  # incorrect
                    react_time_stop = time.time()
                elif wait_keys == 'escape':
                    InitActivity.write(self.cfg_file, self.par_file, self.res_dir, data_file, status='escape')
                    core.quit()

        react_time = react_time_stop - react_time_start - self.trial_dur

        if self.feedback is True:
            feedback = visual.TextStim(self.win,
                                       pos=[0, 0],
                                       height=2.0,
                                       color='black')

            if judge == 1:
                feedback.text = ':)'
            elif judge == 0:
                feedback.text = ':('
            feedback.draw()
            self.win.flip()
            core.wait(0.5)
            fix.draw()
            num.draw()
            self.win.flip()

        return key_press, judge, react_time, trial_time_start

    def run_session(self):
        """
        Run a single session, save data and metadata.

        :return:
        """
        # Check paths
        path = os.path.join(self.res_dir, self.subject)
        if not os.path.exists(path):
            os.makedirs(path)

        # Welcome and wait to start
        welcome = visual.TextStim(self.win,
                                  'Welcome! ' + '\n' + '\n'
                                                       'Your task is to observe the color bar and '
                                                       'judge whether the two dot arrays are in the same sequence as the color bar. ' + '\n' +
                                  'If yes, press the Up arrow; ' + '\n' +
                                  'If not, press the Down arrow. ' + '\n' +
                                  'After giving your response, you can press any key to start the next trial.' + '\n' + '\n' +
                                  'Ready? ' + '\n' +
                                  'Press any key to start this session :)',
                                  color='black',
                                  pos=(0, 0),
                                  height=1.0)
        welcome.draw()
        self.win.mouseVisible = False
        self.win.flip()
        event.waitKeys()

        if feedback is True:
            fbk_msg = visual.TextStim(self.win,
                                      'This is a practice session. ' + '\n' +
                                      'You will get feedback for your response in each trial. ' + '\n' + '\n' +
                                      'If you want to skip the current practice session and continue with the next session, '
                                      'press the key "s" before starting a new trial.' + '\n' +
                                      'Press any key to continue. ',
                                      color='black',
                                      pos=(0, 0),
                                      height=1.0)
            fbk_msg.draw()
            self.win.flip()
            event.waitKeys()

        # Initiate data files
        InitData = write_data.WriteData(self.subject, self.idx, self.res_dir)
        data_file = InitData.head(self.cfg_file, self.par_file)
        InitActivity = write_data.WriteActivity(self.subject, self.idx, self.res_dir)

        count = 0

        # Set the first few trials as warm-up
        warmup = []
        for n in range(self.warmup_nmb):
            for cond in self.conditions:
                warmup.append({'cond': cond, 'width': np.random.randint(5, 8)})
        warmup_stair = data.TrialHandler(warmup, 1, method='sequential')

        for wp in warmup_stair:
            count += 1
            cond = wp['cond']
            width = wp['width']

            patch_xlims = np.array([self.cfg['standard_xlim'], self.cfg['test_xlim']])
            np.random.shuffle(patch_xlims)
            press_key, judge, react_time, trial_time_start = \
                self.run_trial(width, cond, patch_xlims, count, InitActivity, data_file)

            # Save data
            data_dict = {'trial_index': count,
                         'stimulus': cond['stimulus'],
                         'standard_stim': float(cond['standard']),
                         'test_stim': float(cond['test']),
                         'standard_xlim': patch_xlims[0].tolist(),
                         'test_xlim': patch_xlims[1].tolist(),
                         'noise': float(width),
                         'press_key': press_key,
                         'judge': judge,
                         'react_time': react_time,
                         'trial_time_stamp': trial_time_start}

            InitData.write(count, 'warmup', data_dict)

            wait_keys = event.waitKeys()
            if 'escape' in wait_keys:
                InitActivity.write(self.cfg_file, self.par_file, data_file, status='escape')
                core.quit()

        # Run staircase after the warm-up
        stairs = data.MultiStairHandler(stairType='simple',
                                        conditions=self.conditions,
                                        nTrials=self.trial_nmb,
                                        method='random')
        # Pseudo-shuffle: counterbalance the position patterns half to half before running trials
        pos_1_repeat = int(self.trial_nmb / 2)
        pos_2_repeat = self.trial_nmb - pos_1_repeat
        patchpos_1 = np.stack([np.array([self.cfg['standard_xlim'], self.cfg['test_xlim']])] * \
                              pos_1_repeat)
        patchpos_2 = np.stack([np.array([self.cfg['test_xlim'], self.cfg['standard_xlim']])] * \
                              pos_2_repeat)
        patchpos = np.concatenate([patchpos_1, patchpos_2])
        np.random.shuffle(patchpos)

        count_stairs = 0
        for width, cond in stairs:
            count += 1
            count_stairs += 1

            patch_xlims = patchpos[int(np.floor((count_stairs - 1) / len(self.conditions)))]

            press_key, judge, react_time, trial_time_start = \
                self.run_trial(width, cond, patch_xlims, count, InitActivity, data_file)

            # Add intensity-response pairs
            stairs.addResponse(judge, width)

            # Save data
            data_dict = {'trial_index': count,
                         'stimulus': cond['stimulus'],
                         'standard_stim': float(cond['standard']),
                         'test_stim': float(cond['test']),
                         'standard_xlim': patch_xlims[0].tolist(),
                         'test_xlim': patch_xlims[1].tolist(),
                         'noise': float(width),
                         'press_key': press_key,
                         'judge': judge,
                         'react_time': react_time,
                         'trial_time_stamp': trial_time_start}

            InitData.write(count_stairs, 'trial', data_dict)

            wait_keys = event.waitKeys()
            if 'escape' in wait_keys:
                InitActivity.write(self.cfg_file, self.par_file, data_file, status='escape')
                core.quit()

        if self.feedback is True:
            InitActivity.write(self.cfg_file, self.par_file, data_file, status='practice')
        else:
            InitActivity.write(self.cfg_file, self.par_file, data_file, status='completed')


def run_exp(subject, cfg_file, par_files, res_dir, feedback=False):
    """
    Run the experiment by giving inputs

    :param subject:     subject name
    :param cfg_file:    configuration file path
    :param par_files:   a list of parameter file path
    :param res_dir:     the path the store data, DEFAULT as 'data'
    :param feedback:    whether giving feedback on the subject's response each trial, DEFAULT as False
    :return:
    """
    for pidx, pf in enumerate(par_files):

        Exp(subject, cfg_file, pf, res_dir, feedback).run_session()  # run one session

        waitwin = Exp(subject, cfg_file, pf, res_dir, feedback).win

        #  rest between sessions
        if pidx + 1 == len(par_files):
            msg = visual.TextStim(waitwin,
                                  'Well done!' + '\n' +
                                  'You have finished all sessions :)' + '\n' +
                                  'Press any key to quit. ',
                                  color='black',
                                  units='deg',
                                  pos=(0, 0),
                                  height=0.8)
        else:
            msg = visual.TextStim(waitwin,
                                  'Take a break!' + '\n' +
                                  'Then press any key to start the next session :)',
                                  color='black',
                                  units='deg',
                                  pos=(0, 0),
                                  height=0.8)

        msg.draw()
        waitwin.flip()
        event.waitKeys()


# # """test"""
# subject = 'test'
# cfg_file = 'config/test_cfg.yaml'
# par_files = ['config/sub_par/test/noise_HH_135.yaml']
# res_dir = 'data/noise_test'
# feedback = False
# run_exp(subject, cfg_file, par_files, res_dir, feedback)


""" Run experiment from bash """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', help='Subject')
    parser.add_argument('--c', help='Configuration file')
    parser.add_argument('--p', nargs='*', help='Parameter file')
    parser.add_argument('--r', help='Result directory')
    parser.add_argument('--f', type=bool, help='Whether give visual feedback, bool value')

    args = parser.parse_args()
    subject = args.s
    cfg_file = args.c
    par_file = args.p
    results_dir = args.r
    feedback = args.f
    run_exp(subject, cfg_file, par_file, results_dir, feedback)
