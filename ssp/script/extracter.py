#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, August 2011
#
from .. import ar
from .. import core
import numpy as np

# Options
from optparse import OptionParser

def main():
  op = OptionParser()
  op.add_option("-f", dest="fileList", help="List of input output file pairs")
  (opt, arg) = op.parse_args()

  # Fall back on command line input and output
  pairs = []
  if opt.fileList:
      with open(opt.fileList) as f:
          pairs = f.readlines()
  else:
      if len(arg) != 2:
          print "Need two args if no file list"
          exit(1)
      pairs = [ ' '.join(arg) ]

  for pair in pairs:
      loadFile, saveFile = pair.strip().split()

      print "wav: ", loadFile
      pcm = core.PulseCodeModulation()
      a = pcm.WavSource(loadFile)

      # Defaults for 8 kHz
      frameSize = 256
      framePeriod = 80
      lpOrder = 10

      if pcm.rate == 16000:
          frameSize = 400
          framePeriod = 160
          lpOrder = 12

      # Basic preprocessing
      g = np.ndarray((0))
      a = core.ZeroFilter(a)
      f = core.Frame(a, size=frameSize, period=framePeriod, pad=False)
      f = core.Window(f, core.nuttall(frameSize))

      # Next part depends on user
      frontend = core.parameter("FrontEnd", "ar")
      if frontend == "ar":
          a = core.Autocorrelation(f)
          a = core.AutocorrelationAllPassWarp(a, alpha=core.mel[pcm.rate],
                                             size=lpOrder+1)
          a, g = ar.ARLevinson(a, lpOrder)
          #    ridge = Parameter('Ridge', 0.1)
          #    a, g = ARRidge(a, lpOrder, ridge)
          #    a, g = ARLasso(a, lpOrder, ridge)
      elif frontend == "snr":
          a = core.Periodogram(f)
          n = core.Noise(a)
          a = core.SNRSpectrum(a, n * 0.1)
          a = core.Autocorrelation(a, input='psd')
          a, g = ar.ARLevinson(a, lpOrder)
          a = ar.AutocorrelationAllPassWarp(a, alpha=core.mel[pcm.rate],
                                             size=lpOrder+1)
      elif frontend == "sparse":
          a, g = ar.ARSparse(f, lpOrder, core.parameter("Gamma", 1.414))
      elif frontend == "student":
          m = core.AllPassWarpMatrix(frameSize, core.mel[pcm.rate])
          fw = np.dot(f,m.T)
          a, g = ar.ARStudent(fw, lpOrder, core.parameter("DoF", 1.0))
      else:
          print "Unknown front end", frontend

  #    a, g = ar.ARAllPassWarp(a, g, alpha=core.mel[r])

      # Finally, turn the AR coefs into cepstrum
      a = ar.ARCepstrum(a, g, core.parameter("nCepstra", 12))
      m = core.Mean(a)
      a = core.Subtract(a, m)
      m = core.StdDev(a)
      a = core.Divide(a, m)

      print "htk: ", saveFile
      core.HTKSink(saveFile, a, 0.01, "USER")

  return 0
