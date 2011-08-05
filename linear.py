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

# Read wav file
#file = "40jc020c.wav"
file = "FAC_369O5A.wav"

# Load and process
r, a = WavSource(file)
print "rate:", r, "size:", a.size
a = ZeroFilter(a)
f = Frame(a)
print "frame: ", f.shape[0], "x ", f.shape[1]

p = Periodogram(f)
a = Autocorrelation(f)
a, g = ARLevinson(a, order=7)
ls = ARSpectrum(a, g)
wa, wg = BilinearWarpAR(a, g, alpha=0.31)
ws = ARSpectrum(wa, wg)

# Draw it
# fig.add_subplot(2,1,1) # two rows, one column, first plot
frame = 63
fig = plt.figure()
pdfSpec = fig.add_subplot(3,2,1)
pdfPlot = fig.add_subplot(3,2,2)
larSpec = fig.add_subplot(3,2,3)
larPlot = fig.add_subplot(3,2,4)
warSpec = fig.add_subplot(3,2,5)
warPlot = fig.add_subplot(3,2,6)

pdfSpec.imshow(np.rot90(np.log10(p[:,:p.shape[1]/2+1])), aspect='auto')
pdfPlot.plot(np.log10(p[frame,:p.shape[1]/2+1]))
larSpec.imshow(np.rot90(np.log10(ls)), aspect='auto')
larPlot.plot(np.log10(ls[frame]))
warSpec.imshow(np.rot90(np.log10(ws)), aspect='auto')
warPlot.plot(np.log10(ws[frame]))

plt.show()
