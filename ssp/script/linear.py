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

from .. import ar
from .. import core
import numpy as np
import matplotlib.pyplot as plt
lap("Import")

# Load and do basic AR to reconstruct the spectrum
def main():
  pcm = core.PulseCodeModulation()
  wav = pcm.WavSource(file)
  print "File:", file, "rate:", pcm.rate, "size:", wav.size
  if core.parameter("ZF", 0) == 1:
      wav = core.ZeroFilter(wav)
  f = core.Frame(wav, size=256, period=128)
  f = core.Window(f, np.hanning(256))
  print "frame:", f.shape[0], "x", f.shape[1]
  lap("Frame")
  e = core.Energy(f)
  p = core.Periodogram(f)
  lap("Periodogram")
  order = pcm.speech_ar_order()
  a = core.Autocorrelation(f)
  a, g = ar.ARLevinson(a, order)
  lap("Levinson")
  ls = ar.ARSpectrum(a, g, nSpec=128)
  lap("Spectrum")

  # Now do some esoteric AR
  t = core.parameter('AR', 'matrix')
  if t == 'matrix':
      wa, wg = ar.ARMatrix(f, order, method=core.parameter('Method', 'matrix'))
  if t == 'arwarp':
      wa, wg = ar.ARAllPassWarp(a, g, alpha=core.mel[pcm.rate])
  elif t == 'acwarp':
      ac = core.Autocorrelation(f)
      ac = core.AutocorrelationAllPassWarp(ac, alpha=core.mel[pcm.rate],
                                          size=order+1)
      wa, wg = ar.ARLevinson(ac, order)
  elif t == 'tdwarp':
      m = core.AllPassWarpMatrix(256, core.mel[pcm.rate])
      fw = np.dot(f,m.T)
      aw = core.Autocorrelation(fw)
      wa, wg = ar.ARLevinson(aw, order)
  elif t == 'ridge':
      ac = core.Autocorrelation(f)
      wa, wg = ar.ARRidge(ac, order, ridge=0.01)
  elif t == 'lasso':
      ac = core.Autocorrelation(f)
      wa, wg = ar.ARLasso(ac, order, ridge=30)
  elif t == 'sparse':
      wa, wg = ar.ARSparse(f, order, core.parameter("Gamma", 1))
  elif t == 'student':
      wa, wg = ar.ARStudent(f, order, core.parameter("DF", 1))
  lap(t)
  ws = ar.ARSpectrum(wa, wg, nSpec=128)
  lap("Spectrum")

  #llRatio = ARLogLikelihoodRatio(f, order)

  exn = ar.ARExcitation(f, a, g)
  exnw = ar.ARExcitation(f, wa, wg)


  # Draw it
  # fig.add_subplot(2,1,1) # two rows, one column, first plot
  frame = core.parameter('Frame', 0)
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

  return 0
