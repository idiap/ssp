#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, August 2011
#
from ssp import *

# Options
from optparse import OptionParser
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

    r, a = WavSource(loadFile)
    a = ZeroFilter(a)
    f = Frame(a, size=256, period=80)
    a = Autocorrelation(f)
    a, g = ARLevinson(a, order=10)
    a, g = ARBilinearWarp(a, g, alpha=mel[r])
    a = ARCepstrum(a, g)
    HTKSink(saveFile, a, 0.01, "USER")
