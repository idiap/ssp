#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#

# Python 3:
#import argparse
#ap = argparse.ArgumentParser('Autoregressive modelling')
#ap.add_argument('file')
#arg = ap.parse_args()

# Python 2:
from optparse import OptionParser
op = OptionParser()
(option, arg) = op.parse_args()
if (len(arg) < 1):
    print "Need one arg"
    exit(1)
file = arg[0]

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
import numpy as np
import matplotlib.pyplot as plt
lap("Import")

# Load and process
print "Using file:", file
r, a = WavSource(file)
print "rate:", r, "size:", a.size
a = ZeroFilter(a)
f = Frame(a, size=256, period=128)
f = Window(f, np.hanning(256))
print "frame:", f.shape[0], "x", f.shape[1]
lap("Frame")

e = Energy(f)
p = Periodogram(f)
lap("Periodogram")

order = 10
if 1:
    a = Autocorrelation(f)
    a, g = ARLevinson(a, order)
    lap("Levinson")
else:
    a, g = ARMatrix(f, order, method='acmatrix')
    lap("Matrix")

ls = ARSpectrum(a, g, nSpec=128)
lap("Spectrum")

t = Parameter('LP', 'arwarp')
if t == 'arwarp':
    wa, wg = ARBilinearWarp(a, g, alpha=mel[r])
    lap("AR Warp")
elif t == 'acwarp':
    ac = Autocorrelation(f)
    ac = AutocorrelationBilinearWarp(ac, alpha=mel[r], size=order+1)
    wa, wg = ARLevinson(ac, order)
#    wa, wg = ARRidge(ac, order, ridge=0.1)
#    wa, wg = ARLasso(ac, order, ridge=10)
    lap("AC Warp")
elif t == 'ridge':
    ac = Autocorrelation(f)
    wa, wg = ARRidge(ac, order, ridge=0.01)
    lap("Ridge")
elif t == 'lasso':
    ac = Autocorrelation(f)
    wa, wg = ARLasso(ac, order, ridge=30)
    lap("Lasso")
elif t == 'sparse':
    wa, wg = ARSparse(f, order)
    lap("Sparse")
ws = ARSpectrum(wa, wg, nSpec=128)
lap("Spectrum")

#llRatio = ARLogLikelihoodRatio(f, order)

exn = ARExcitation(f, a, g)
exnw = ARExcitation(f, wa, wg)

# Draw it
# fig.add_subplot(2,1,1) # two rows, one column, first plot
frame = 13
fig = plt.figure()
plt.bone()
pdfSpec = fig.add_subplot(3,2,1)
pdfPlot = fig.add_subplot(3,2,2)
larSpec = fig.add_subplot(3,2,3)
larPlot = fig.add_subplot(3,2,4)
warSpec = fig.add_subplot(3,2,5)
warPlot = fig.add_subplot(3,2,6)

pdfSpec.imshow(np.transpose(np.log10(p[:,:p.shape[1]/2+1])),
               origin='lower', aspect='auto')
pdfPlot.plot(np.log10(e/f.shape[1]))
pdfPlot.plot(np.log10(g))
pdfPlot.plot(np.log10(wg))

#pdfPlot.plot(llRatio)

larSpec.imshow(np.transpose(np.log10(ls)), origin='lower', aspect='auto')
#larPlot.plot(Norm(a, 1))
#larPlot.plot(Norm(wa, 1))
larPlot.plot(exn[frame])
larPlot.plot(exn[frame])
larPlot.plot(exnw[frame])


warSpec.imshow(np.transpose(np.log10(ws)), origin='lower', aspect='auto')
warPlot.plot(np.log10(p[frame,:p.shape[1]/2+1]/f.shape[1]))
warPlot.plot(np.log10(ls[frame]), label="Linear")
warPlot.plot(np.log10(ws[frame]), label="Warped")
#warPlot.legend()

plt.show()
