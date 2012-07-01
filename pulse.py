#!/usr/bin/env python2
#
# Copyright 2012 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, July 2012
#
from ssp import *

fig = Figure(2,1)
ax1 = fig.subplot()
ax2 = fig.subplot()

period = 200
x = np.arange(period)
#for t in ('trig', 'poly', 'gamma', 'igamma', 'lf'):
for t in ('trig', 'lf'):
    if t != 'lf':
        ax1.plot(x, pulse(period, t, derivative=False))
    ax2.plot(x, pulse(period, t))

plt.show()
