# !/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Tool lib for curve fitting and modeling.

Created on 02.09.20

@author yannansu
"""
from __future__ import absolute_import, division, print_function
from builtins import object
import numpy as np
from scipy import optimize
import lmfit
import matplotlib.pyplot as plt
from scipy.stats import vonmises
import scipy.stats as st
from scipy.special import i0


class _baseFunctionFit(object):
    """
    =================================================================================
    Fit PF to cumulative Gaussian and include all parameter fitting (lapse, chance).
    Adapted from Psychopy.data.FitCumNormal.
    =================================================================================

    Not needed by most users except as a superclass for developing
    your own functions

    Derived classes must have _eval and _inverse methods with @staticmethods
    """

    def __init__(self, xx, yy, sems=1.0, guess=None, display=1,
                 expectedMin=0.5, lapse=0.05, optimize_kws=None):
        super(_baseFunctionFit, self).__init__()
        self.xx = np.array(xx)
        self.yy = np.array(yy)
        self.sems = np.array(sems)
        if not hasattr(sems, "__len__"):
            # annoyingly in numpy 1.13 len(numpy.array(1)) gives an error
            self.sems.shape = (1,)  # otherwise we can't get len (in numpy 1.13)
        self.expectedMin = expectedMin
        self.lapse = lapse
        self.guess = guess
        self.optimize_kws = {}
        if optimize_kws is not None:
            self.optimize_kws = optimize_kws
        # for holding error calculations:
        self.ssq = 0
        self.rms = 0
        self.chi = 0
        # do the calculations:
        self._doFit()

    def _doFit(self):
        """The Fit class that derives this needs to specify its _evalFunction
        """
        # get some useful variables to help choose starting fit vals
        # self.params = optimize.fmin_powell(self._getErr, self.params,
        #    (self.xx,self.yy,self.sems),disp=self.display)
        # self.params = optimize.fmin_bfgs(self._getErr, self.params, None,
        #    (self.xx,self.yy,self.sems),disp=self.display)
        from scipy import optimize
        # don't import optimize at top of script. Slow and not always present!

        global _chance
        global _lapse
        _chance = self.expectedMin
        _lapse = self.lapse
        if len(self.sems) == 1:
            sems = None
        else:
            sems = self.sems
        self.params, self.covar = optimize.curve_fit(
            self._eval, self.xx, self.yy, p0=self.guess, sigma=sems,
            **self.optimize_kws)
        self.ssq = self._getErr(self.params, self.xx, self.yy, 1.0)
        self.chi = self._getErr(self.params, self.xx, self.yy, self.sems)
        self.rms = self.ssq / len(self.xx)

    def _getErr(self, params, xx, yy, sems):
        mod = self.eval(xx, params)
        err = sum((yy - mod) ** 2 / sems)
        return err

    def eval(self, xx, params=None):
        """Evaluate xx for the current parameters of the model, or for
        arbitrary params if these are given.
        """
        if params is None:
            params = self.params
        global _chance
        global _lapse
        _chance = self.expectedMin
        _lapse = self.lapse
        # _eval is a static method - must be done this way because the
        # curve_fit function doesn't want to have any `self` object as
        # first arg
        yy = self._eval(xx, *params)
        return yy

    def inverse(self, yy, params=None):
        """Evaluate yy for the current parameters of the model,
        or for arbitrary params if these are given.
        """
        if params is None:
            # so the user can set params for this particular inv
            params = self.params
        xx = self._inverse(yy, *params)
        return xx


class FitCumNormal(_baseFunctionFit):
    """
    =================================================================================
    Fit PF to cumulative Gaussian and include all parameter fitting (lapse, chance).
    Adapted from Psychopy.data.FitCumNormal.
    =================================================================================
    Fit a Cumulative Normal function (aka error function or erf)
    of the form::

        y = chance + (1-chance-lapse)*((special.erf((xx-xShift)/(sqrt(2)*sd))+1)*0.5)

    and with inverse::

        x = xShift+sqrt(2)*sd*(erfinv(((yy-chance)/(1-chance-lapse)-.5)*2))

    After fitting the function you can evaluate an array of x-values
    with fit.eval(x), retrieve the inverse of the function with
    fit.inverse(y) or retrieve the parameters from fit.params (a list
    with [centre, sd] for the Gaussian distribution forming the cumulative)

    NB: Prior to version 1.74 the parameters had different meaning, relating
    to xShift and slope of the function (similar to 1/sd). Although that is
    more in with the parameters for the Weibull fit, for instance, it is less
    in keeping with standard expectations of normal (Gaussian distributions)
    so in version 1.74.00 the parameters became the [centre,sd] of the normal
    distribution.

    """

    # static methods have no `self` and this is important for
    # optimise.curve_fit
    @staticmethod
    def _eval(xx, xShift, sd):
        from scipy import special
        global _chance
        global _lpase
        xx = np.asarray(xx)
        # NB np.special.erf() goes from -1:1
        yy = (_chance + (1 - _chance - _lapse) *
              ((special.erf((xx - xShift) / (np.sqrt(2) * sd)) + 1) * 0.5))
        return yy

    @staticmethod
    def _inverse(yy, xShift, sd):
        from scipy import special
        global _chance
        global _lpase
        yy = np.asarray(yy)
        # xx = (special.erfinv((yy-chance)/(1-chance)*2.0-1)+xShift)/xScale
        # NB: np.special.erfinv() goes from -1:1
        xx = (xShift + np.sqrt(2) * sd *
              special.erfinv(((yy - _chance) / (1 - _chance - _lapse) - 0.5) * 2))
        return xx

