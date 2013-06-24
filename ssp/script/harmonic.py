#!/usr/bin/env python2
#
# Copyright 2012 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, July 2012
#
from .. import ar
from .. import core
import numpy as np
import matplotlib.pyplot as plt

r = 1000

a1 = ar.ARHarmonicPoly(21, r, 0.99)
a2 = ar.ARHarmonicPoly(32, r, 0.99)
s1 = ar.ARSpectrum(a1, 1.0, 10010)
s2 = ar.ARSpectrum(a2, 1.0, 10010)

fig = core.Figure(1,1)
ax = fig.subplot()
ax.plot(np.log(s1))
ax.plot(np.log(s2))
plt.show()
