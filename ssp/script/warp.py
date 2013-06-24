#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#

from .. import core

import numpy as np
import matplotlib.pyplot as plt

def main():
  i = np.identity(30)
  o1 = core.AllPassWarpOppenheim(i, alpha=0, size=40)
  o2 = core.AllPassWarpOppenheim(i, alpha=0.1, size=40)
  o3 = core.AllPassWarpOppenheim(i, alpha=0.3, size=40)
  #o5 = core.AllPassWarpOppenheim(i, alpha=-0.1, size=40)
  #o6 = core.AllPassWarpOppenheim(i, alpha=-0.3, size=40)
  o5 = core.AllPassWarpMatrix(30, alpha=-0.1, size=40)
  o6 = core.AllPassWarpMatrix(30, alpha=-0.3, size=40)


  #m = core.AllPassWarpMatrix(4, alpha=0.42, size=40)

  fig = plt.figure()
  o1Mat = fig.add_subplot(2,3,1)
  o2Mat = fig.add_subplot(2,3,2)
  o3Mat = fig.add_subplot(2,3,3)
  o5Mat = fig.add_subplot(2,3,5)
  o6Mat = fig.add_subplot(2,3,6)

  o1Mat.imshow(o1.T)
  o2Mat.imshow(o2.T)
  o3Mat.imshow(o3.T)
  o5Mat.imshow(o5)
  o6Mat.imshow(o6)

  plt.show()
  
  return 0
