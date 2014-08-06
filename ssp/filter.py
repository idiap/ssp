#!/usr/bin/python2
#
# Copyright 2014 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, June 2014
#

import numpy as np
import scipy.signal as sp

class Filter:
    """
    Linear filter class where the poles and zeros can be added individually.
    """
    def __init__(self):
        self.b = np.ndarray((0))
        self.a = np.ndarray((0))
        self.pole = np.ndarray((0))
        self.zero = np.ndarray((0))
        self.solved = False

    def addPole(self, z):
        self.pole = np.append(self.pole, z)
        self.solved = False

    def addZero(self, z):
        self.zero = np.append(self.zero, z)
        self.solved = False

    def clear(self):
        self.pole.resize((0))
        self.zero.resize((0))
        self.solved = False

    def addConjugatePole(self, z, arg=None):
        # If an angle is given, assume it's a polar value and convert
        if arg:
            z = abs(z)*(np.cos(arg) + np.sin(arg)*1j)
        self.pole = np.append(self.pole, [z, z.conjugate()])
        self.solved = False

    def addConjugateZero(self, z, arg=None):
        # If an angle is given, assume it's a polar value and convert
        if arg:
            z = abs(z)*(np.cos(arg) + np.sin(arg)*1j)
        self.zero = np.append(self.zero, [z, z.conjugate()])
        self.solved = False

    def solve(self):
        if (len(self.pole) == 0):
            self.a = np.ones((1))
        else:
            self.a = np.poly(self.pole)
        if (len(self.zero) == 0):
            self.b = np.ones((1))
        else:
            self.b = np.poly(self.zero)
        self.solved = True

    def filter(self, a):
        if not self.solved:
            self.solve()
        y = sp.lfilter(self.b, self.a, a)
        return y
