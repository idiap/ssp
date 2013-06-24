#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#
from optparse import OptionParser
op = OptionParser()
(option, arg) = op.parse_args()
if (len(arg) < 1):
    print "Need one arg"
    exit(1)
file = arg[0]

import ssp
import numpy as np
import matplotlib.pyplot as plt

# Load and process
pcm = ssp.PulseCodeModulation()
a = pcm.WavSource(file)
a = ssp.ZeroFilter(a)
f = ssp.Frame(a, size=256, period=80)
type = ssp.parameter('Type', 'ar')
if type == 'psd':
    p = ssp.Periodogram(f)
    p = p[:,:p.shape[1]/2+1]
elif type == 'ar':
    a = ssp.Autocorrelation(f)
    a, g = ssp.ARLevinson(a, pcm.speech_ar_order())
    p = ssp.ARSpectrum(a, g, nSpec=64)
elif type == 'snr':
    p = ssp.Periodogram(f)
    n = ssp.Noise(p)
    p = ssp.SNRSpectrum(p, n)
    p = p[:,:p.shape[1]/2+1]

# Draw it
plt.bone()
plt.yticks((0,63), ('0', 'fs/2'))
plt.imshow(np.transpose(np.log10(p)), origin='lower')
plt.show()
