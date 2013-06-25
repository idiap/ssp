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

#from optparse import OptionParser
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments(command_line_parameters):
  """Defines the command line parameters that are accepted."""

  # create parser
  parser = argparse.ArgumentParser(description='Computes the spectrogram of an audio signal (wav file only)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('filename', type=file, help='Path to the file to process.')
  parser.add_argument('-t', '--type', type=str, choices=('ar', 'psd', 'snr'), default='ar', help='The type of spectrogram to generate')
  parser.add_argument('-n', '--no-show', action='store_true', help='Do not call the show() method at the end of the script (mostly for testing purpose)')
  return parser.parse_args(command_line_parameters)


def spectrogram(args):

  # Load and process
  pcm = core.PulseCodeModulation()
  a = pcm.WavSource(args.filename)
  a = core.ZeroFilter(a)
  f = core.Frame(a, size=256, period=80)
  ptype = core.parameter('Type', args.type)
  if ptype == 'psd':
    p = core.Periodogram(f)
    p = p[:,:p.shape[1]/2+1]
  elif ptype == 'ar':
    a = core.Autocorrelation(f)
    a, g = ar.ARLevinson(a, pcm.speech_ar_order())
    p = ar.ARSpectrum(a, g, nSpec=64)
  elif ptype == 'snr':
    p = core.Periodogram(f)
    n = core.Noise(p)
    p = core.SNRSpectrum(p, n)
    p = p[:,:p.shape[1]/2+1]
  else:
    raise runtime_error("Unsupported type for the spectrogram")

  # Draw it
  plt.bone()
  plt.yticks((0,63), ('0', 'fs/2'))
  plt.imshow(np.transpose(np.log10(p)), origin='lower')
  if args.no_show == False:
    plt.show()

  return 0

def main(command_line_parameters = sys.argv):
  """Executes the main function"""
  # do the command line parsing
  args = parse_arguments(command_line_parameters[1:])

  # perform face verification test
  return spectrogram(args)

if __name__ == "__main__":
  main()
