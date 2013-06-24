#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#
import matplotlib
import numpy as np
from matplotlib.pyplot import figure, show, rc, grid

def main():
  # Force square figure; looks better for polar
  width, height = matplotlib.rcParams['figure.figsize']
  size = min(width, height)
  fig = figure(figsize=(size, size))

  # Axes
  ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

  # Circle
  arg = np.arange(0, 2 * np.pi, np.pi/100)
  magL2 = np.ones(arg.size)
  ax.plot(arg, magL2)

  # L1 in L2
  c = np.cos(arg)
  s = np.sin(arg)
  magL1 = np.abs(c) + np.abs(s)
  ax.plot(arg, magL1)

  # L2 / L1
  magL21 = magL2 / magL1
  ax.plot(arg, magL21)

  # Sum to 1
  #magSum = magL2 / ()

  ax.set_rmax(2.0)
  show()

  return 0
