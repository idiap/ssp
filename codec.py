#!/usr/bin/env python2
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

# Overlap add
ola = True
#ola = False

for pair in pairs:
    loadFile, saveFile = pair.strip().split()
    print "wav: ", loadFile
    r, a = WavSource(loadFile)

    # Defaults for 8 kHz
    if r == 8000:
        frameSize = 256
        framePeriod = 256
        lpOrder = 10
    elif r == 16000:
        framePeriod = 256
        frameSize = 512
        pitchSize = 1024
        lpOrder = 20
    else:
        exit

    if ola:
        synthSize = framePeriod * 2
    else:
        synthSize = framePeriod
        
    f = Frame(a, size=frameSize, period=framePeriod)
    f = ZeroMean(f)
    aw = np.hanning(frameSize+1)
    aw = np.delete(aw, -1)
    w = Window(f, aw)
    ac = Autocorrelation(w)
    lp = Parameter('AR', 'levinson')
    if lp == 'levinson':
        ar, g = ARLevinson(ac, lpOrder)
    elif lp == 'ridge':
        ar, g = ARRidge(ac, lpOrder, 0.03)
    elif lp == 'lasso':
        ar, g = ARLasso(ac, lpOrder, 10)
    elif lp == 'sparse':
        ar, g = ARSparse(f, lpOrder)

    # The pitch window should be longer with no sidelobes
    pf = Frame(a, size=pitchSize, period=framePeriod)
    pitch, hnr = ACPitch(pf)

    ex = Parameter('Excitation', 'synth')
    if ex == 'ar':
        e = ARExcitation(f, ar, g)
    elif ex == 'noise':
        e = np.random.normal(size=f.shape)
    elif ex == 'robot':
        ew = np.zeros(len(a))
        period = int(1.0 / 200 * r)
        for i in range(0, len(ew), period):
            ew[i] = 1.0
        e = Frame(ew, size=synthSize, period=framePeriod)        
    elif ex == 'synth':
        # Harmonic part
        h = np.zeros(len(a))
        i = 0
        frame = 0
        while i < len(a) and frame < len(pitch):
            period = int(1.0 / pitch[frame] * r)
            #period = int(1.0 / 200 * r)
            weight = np.sqrt(hnr[frame] / (hnr[frame] + 1) * period)
            h[i] = weight
            i += period
            frame = i // framePeriod
        fh = Frame(h, size=synthSize, period=framePeriod)

        # Noise part
        n = np.random.normal(size=len(a))
        fn = Frame(n, size=synthSize, period=framePeriod)
        for i in range(len(fn)):
            fn[i] *= np.sqrt(1.0 / (hnr[i] + 1))
        e = fn + fh*3

    sw = np.hanning(synthSize+1)
    sw = np.delete(sw, -1)

    #e = Window(e, sw)
    s = ARResynthesis(e, ar, g)
    print "wav: ", saveFile
    if ola:
        s = Window(s, sw)
        s = OverlapAdd(s)
        WavSink(s.flatten('C'), saveFile, r)
    else:
        WavSink(e.flatten('C') / 1000, saveFile, r)

if False:
    fig = Figure(5, 1)
    #stddev = np.sqrt(kVar)
    sPlot = fig.subplot()
    sPlot.plot(pitch, 'c')
    #sPlot.plot(kPitch + stddev, 'b')
    #sPlot.plot(kPitch - stddev, 'b')
    sPlot.set_xlim(0, len(pitch))
    sPlot.set_ylim(0, 500)
    plt.show()
