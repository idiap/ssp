#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, August 2011
#
from ... import ssp
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
      pcm = ssp.PulseCodeModulation()
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
      a = ssp.ZeroFilter(a)
      f = ssp.Frame(a, size=frameSize, period=framePeriod, pad=False)
      f = ssp.Window(f, ssp.nuttall(frameSize))

      # Next part depends on user
      frontend = ssp.parameter("FrontEnd", "ar")
      if frontend == "ar":
          a = ssp.Autocorrelation(f)
          a = ssp.AutocorrelationAllPassWarp(a, alpha=ssp.mel[pcm.rate],
                                             size=lpOrder+1)
          a, g = ssp.ARLevinson(a, lpOrder)
          #    ridge = Parameter('Ridge', 0.1)
          #    a, g = ARRidge(a, lpOrder, ridge)
          #    a, g = ARLasso(a, lpOrder, ridge)
      elif frontend == "snr":
          a = ssp.Periodogram(f)
          n = ssp.Noise(a)
          a = ssp.SNRSpectrum(a, n * 0.1)
          a = ssp.Autocorrelation(a, input='psd')
          a, g = ssp.ARLevinson(a, lpOrder)
          a = ssp.AutocorrelationAllPassWarp(a, alpha=ssp.mel[pcm.rate],
                                             size=lpOrder+1)
      elif frontend == "sparse":
          a, g = ssp.ARSparse(f, lpOrder, ssp.parameter("Gamma", 1.414))
      elif frontend == "student":
          m = ssp.AllPassWarpMatrix(frameSize, ssp.mel[pcm.rate])
          fw = np.dot(f,m.T)
          a, g = ssp.ARStudent(fw, lpOrder, ssp.parameter("DoF", 1.0))
      else:
          print "Unknown front end", frontend

  #    a, g = ARAllPassWarp(a, g, alpha=mel[r])

      # Finally, turn the AR coefs into cepstrum
      a = ssp.ARCepstrum(a, g, ssp.parameter("nCepstra", 12))
      m = ssp.Mean(a)
      a = ssp.Subtract(a, m)
      m = ssp.StdDev(a)
      a = ssp.Divide(a, m)

      print "htk: ", saveFile
      ssp.HTKSink(saveFile, a, 0.01, "USER")


