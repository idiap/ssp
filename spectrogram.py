#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#
from ssp import *
import numpy as np
import matplotlib.pyplot as plt

# Load and process
file = "FAC_13A.wav"
r, a = WavSource(file)
a = ZeroFilter(a)
f = Frame(a, size=256, period=128)
if False:
    p = Periodogram(f)
    p = p[:,:p.shape[1]/2+1]
else:
    a = Autocorrelation(f)
    a, g = ARLevinson(a, 10)
    p = ARSpectrum(a, g, nSpec=64)

# Draw it
plt.bone()
plt.yticks((0,63), ('0', 'fs/2'))
plt.imshow(np.transpose(np.log10(p)), origin='lower')
plt.show()
