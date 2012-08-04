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
import numpy as np

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

    print "wav: ", loadFile
    r, a = WavSource(loadFile)

    # Defaults for 8 kHz
    frameSize = 256
    framePeriod = 80
    lpOrder = 10

    if r == 16000:
        frameSize = 400
        framePeriod = 160
        lpOrder = 12
        
    a = ZeroFilter(a)
    f = Frame(a, size=frameSize, period=framePeriod)
    f = Window(f, nuttall(frameSize))
    if 1:
        a = Autocorrelation(f)
    else:
        a = Periodogram(f)
        n = Noise(a)
        a = SNRSpectrum(a, n * 0.1)
        a = Autocorrelation(a, input='psd')
    a = AutocorrelationAllPassWarp(a, alpha=mel[r], size=lpOrder+1)

    ridge = Parameter('Ridge', 0.1)
#    a, g = ARRidge(a, lpOrder, ridge)
#    a, g = ARLasso(a, lpOrder, ridge)
    a, g = ARSparse(a, lpOrder)
#    a, g = ARLevinson(a, lpOrder)
#    a, g = ARAllPassWarp(a, g, alpha=mel[r])
    a = ARCepstrum(a, g, Parameter("nCepstra", 12))
    m = Mean(a)
    a = Subtract(a, m)
    m = StdDev(a)
    a = Divide(a, m)

    print "htk: ", saveFile
    HTKSink(saveFile, a, 0.01, "USER")
