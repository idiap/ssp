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

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_arguments(command_line_parameters):
  """Defines the command line parameters that are accepted."""

  # create parser
  parser = argparse.ArgumentParser(description='Warps (TODO: add documentation)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-n', '--no-show', action='store_true', help='Do not call the show() method at the end of the script (mostly for testing purpose)')
  return parser.parse_args(command_line_parameters)

def warp(args):

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
 
  if args.no_show == False:
    plt.show()
  
  return 0

def main(command_line_parameters = sys.argv):
  """Executes the main function"""
  # do the command line parsing
  args = parse_arguments(command_line_parameters[1:])

  # perform face verification test
  return warp(args)

if __name__ == "__main__":
  main()
