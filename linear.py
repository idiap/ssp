#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#
import argparse
ap = argparse.ArgumentParser('Autoregressive modelling')
ap.add_argument('file')
arg = ap.parse_args()

import time

# Timer
ti = time.clock()
def lap(func):
  global ti
  now = time.clock()
  elapsed = now-ti
  ti = now
  print func, elapsed

from ssp import *
lap("ssp")
import numpy as np
lap("numpy")
import matplotlib.pyplot as plt
lap("plt")

# Load and process
print "Using file:", arg.file
r, a = WavSource(arg.file)
print "rate:", r, "size:", a.size
a = ZeroFilter(a)
f = Frame(a)
print "frame: ", f.shape[0], "x ", f.shape[1]
lap("Frame")

p = Periodogram(f)
lap("Periodogram")
a = Autocorrelation(f)
lap("AC")
a, g = ARLevinson(a, 7)
lap("Levinson")
ls = ARSpectrum(a, g)
lap("Spectrum")
wa, wg = ARBilinearWarp(a, g, alpha=0.31)
lap("Warp")
ws = ARSpectrum(wa, wg)
lap("Spectrum")

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
