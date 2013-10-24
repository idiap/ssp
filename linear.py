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

import ssp
import numpy as np
import matplotlib.pyplot as plt
lap("Import")

# Load and do basic AR to reconstruct the spectrum
pcm = ssp.PulseCodeModulation()
wav = pcm.WavSource(file)
print "File:", file, "rate:", pcm.rate, "size:", wav.size
if ssp.parameter("ZF", 0) == 1:
    wav = ssp.ZeroFilter(wav)
f = ssp.Frame(wav, size=256, period=128)
f = ssp.Window(f, np.hanning(256))
print "frame:", f.shape[0], "x", f.shape[1]
lap("Frame")
e = ssp.Energy(f)
p = ssp.Periodogram(f)
lap("Periodogram")
order = pcm.speech_ar_order()
a = ssp.Autocorrelation(f)
a, g = ssp.ARLevinson(a, order)
lap("Levinson")
ls = ssp.ARSpectrum(a, g, nSpec=128)
lap("Spectrum")

# Now do some esoteric AR
t = ssp.parameter('AR', 'matrix')
if t == 'matrix':
    wa, wg = ssp.ARMatrix(f, order, method=ssp.parameter('Method', 'matrix'))
if t == 'arwarp':
    wa, wg = ssp.ARAllPassWarp(a, g, alpha=ssp.mel[pcm.rate])
elif t == 'acwarp':
    ac = ssp.Autocorrelation(f)
    ac = ssp.AutocorrelationAllPassWarp(ac, alpha=ssp.mel[pcm.rate],
                                        size=order+1)
    wa, wg = ssp.ARLevinson(ac, order)
elif t == 'tdwarp':
    m = ssp.AllPassWarpMatrix(256, ssp.mel[pcm.rate])
    fw = np.dot(f,m.T)
    aw = ssp.Autocorrelation(fw)
    wa, wg = ssp.ARLevinson(aw, order)
elif t == 'ridge':
    ac = ssp.Autocorrelation(f)
    wa, wg = ssp.ARRidge(ac, order, ridge=0.01)
elif t == 'lasso':
    ac = ssp.Autocorrelation(f)
    wa, wg = ssp.ARLasso(ac, order, ridge=30)
elif t == 'sparse':
    wa, wg = ssp.ARSparse(f, order, ssp.parameter("Gamma", 1))
elif t == 'student':
    wa, wg = ssp.ARStudent(f, order, ssp.parameter("DF", 1))
lap(t)
ws = ssp.ARSpectrum(wa, wg, nSpec=128)
lap("Spectrum")

#llRatio = ARLogLikelihoodRatio(f, order)

exn = ssp.ARExcitation(f, a, g)
exnw = ssp.ARExcitation(f, wa, wg)


# Draw it
# fig.add_subplot(2,1,1) # two rows, one column, first plot
frame = ssp.parameter('Frame', 0)
fig = plt.figure()
plt.bone()
pdfSpec = fig.add_subplot(3,2,1)
pdfPlot = fig.add_subplot(3,2,2)
larSpec = fig.add_subplot(3,2,3)
larPlot = fig.add_subplot(3,2,4)
warSpec = fig.add_subplot(3,2,5)
warPlot = fig.add_subplot(3,2,6)

pdfSpec.imshow(np.transpose(np.log10(p)), origin='lower', aspect='auto')
pdfPlot.plot(np.log10(e/f.shape[1]))
pdfPlot.plot(np.log10(g))
pdfPlot.plot(np.log10(wg))
pdfPlot.set_xlim(0, len(g))

#pdfPlot.plot(llRatio)

larSpec.imshow(np.transpose(np.log10(ls)), origin='lower', aspect='auto')
#larPlot.plot(Norm(a, 1))
#larPlot.plot(Norm(wa, 1))
larPlot.plot(exn[frame])
larPlot.plot(f[frame] / np.sqrt(g[frame]))
larPlot.plot(exnw[frame])
larPlot.set_xlim(0, f.shape[1])


warSpec.imshow(np.transpose(np.log10(ws)), origin='lower', aspect='auto')
warPlot.plot(np.log10(p[frame]/f.shape[1]))
warPlot.plot(np.log10(ls[frame]), label="Linear")
warPlot.plot(np.log10(ws[frame]), label="Warped")
warPlot.set_xlim(0, ls.shape[1])
#warPlot.legend()

plt.show()
