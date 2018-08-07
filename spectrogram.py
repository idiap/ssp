#!/usr/bin/env python3
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
    print("Need one arg")
    exit(1)
file = arg[0]

import ssp
import numpy as np

# Load and process
pcm = ssp.PulseCodeModulation()
a = pcm.WavSource(file)
if (ssp.parameter('Pre', None)):
    a = ssp.ZeroFilter(a)
framePeriod = pcm.seconds_to_period(0.01)
frameSize = pcm.seconds_to_period(0.02, 'atleast')
f = ssp.Frame(a, size=frameSize, period=framePeriod)
w = ssp.nuttall(frameSize+1)
w = np.delete(w, -1)
wf = ssp.Window(f, w)
type = ssp.parameter('Type', 'psd')
if type == 'psd':
    p = ssp.Periodogram(wf)
    p = p[:,:p.shape[1]//2+1]
elif type == 'ar':
    a = ssp.Autocorrelation(wf)
    a, g = ssp.ARLevinson(a, pcm.speech_ar_order())
    p = ssp.ARSpectrum(a, g, nSpec=128)
elif type == 'snr':
    p = ssp.Periodogram(wf)
    n = ssp.Noise(p)
    p = ssp.SNRSpectrum(p, n)
    p = p[:,:p.shape[1]/2+1]

# Draw it
fig = ssp.Figure(2, 1)
p1 = fig.SpectrumPlot(p, pcm)
p2 = fig.EnergyPlot(f, pcm)
fig.show()
