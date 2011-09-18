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
file = "FAC_369O5A.wav"
r, a = WavSource(file)
a = ZeroFilter(a)
f = Frame(a, size=256, period=80)
type = 'ar'
if type == 'psd':
    p = Periodogram(f)
    p = p[:,:p.shape[1]/2+1]
elif type == 'ar':
    a = Autocorrelation(f)
    a, g = ARLevinson(a, 10)
    p = ARSpectrum(a, g, nSpec=64)
elif type == 'snr':
    p = Periodogram(f)
    n = Noise(p)
    p = SNRSpectrum(p, n)
    p = p[:,:p.shape[1]/2+1]    

# Draw it
plt.bone()
plt.yticks((0,63), ('0', 'fs/2'))
plt.imshow(np.transpose(np.log10(p)), origin='lower')
plt.show()
