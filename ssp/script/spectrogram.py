#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#

from .. import ar
from .. import core

from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt

def main():
  op = OptionParser()
  (option, arg) = op.parse_args()
  if (len(arg) < 1):
      print "Need one arg"
      exit(1)
  file = arg[0]

  # Load and process
  pcm = core.PulseCodeModulation()
  a = pcm.WavSource(file)
  a = core.ZeroFilter(a)
  f = core.Frame(a, size=256, period=80)
  type = core.parameter('Type', 'ar')
  if type == 'psd':
      p = core.Periodogram(f)
      p = p[:,:p.shape[1]/2+1]
  elif type == 'ar':
      a = core.Autocorrelation(f)
      a, g = ar.ARLevinson(a, pcm.speech_ar_order())
      p = ar.ARSpectrum(a, g, nSpec=64)
  elif type == 'snr':
      p = core.Periodogram(f)
      n = core.Noise(p)
      p = core.SNRSpectrum(p, n)
      p = p[:,:p.shape[1]/2+1]

  # Draw it
  plt.bone()
  plt.yticks((0,63), ('0', 'fs/2'))
  plt.imshow(np.transpose(np.log10(p)), origin='lower')
  plt.show()

  return 0
