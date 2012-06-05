#!/usr/bin/python2
#
# Copyright 2012 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, May 2012
#
from optparse import OptionParser
op = OptionParser()
op.add_option("-f", dest="fileList", help="List of input output file pairs")
(opt, arg) = op.parse_args()

# SSP import
from ssp import *

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
    r, a = WavSource(loadFile)

    # Defaults for 8 kHz
    frameSize = 256
    framePeriod = 256
    lpOrder = 10

    if r == 16000:
        frameSize = 512
        framePeriod = 512
        lpOrder = 18
        
#    a = ZeroFilter(a)
    f = Frame(a, size=frameSize, period=framePeriod)
    w = Window(f, nuttall(frameSize))
    ac = Autocorrelation(w)
    ar, g = ARLevinson(ac, lpOrder)
    e = ARExcitation(f, ar, g)
    # e = np.random.normal(size=f.shape)

    s = ARResynthesis(e, ar, g)
    print "wav: ", saveFile
    e /= np.amax(e)
    WavSink(e.flatten('A'), saveFile, r)
