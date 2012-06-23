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
        lpOrder = 24
    if ola:
        framePeriod /= 2
        
    f = Frame(a, size=frameSize, period=framePeriod)
    hw = np.hanning(frameSize+1)
    hw = np.delete(hw, -1)
    w = Window(f, hw)
    ac = Autocorrelation(w)
    ar, g = ARLevinson(ac, lpOrder)

    # The pitch window should be longer with no sidelobes
    pf = Frame(a, size=1024, period=framePeriod)
    pitch, hnr = ACPitch(pf)

    ex = Parameter('Excitation', 'synth')
    if ex == 'ar':
        e = ARExcitation(f, ar, g)
    elif ex == 'random':
        e = np.random.normal(size=f.shape)
    elif ex == 'robot':
        ew = np.zeros(a.shape)
        period = int(1.0 / 200 * r)
        for i in range(0, len(ew), period):
            ew[i] = 1.0
        e = Frame(ew, size=frameSize, period=framePeriod)        
    elif ex == 'synth':
        ew = np.zeros(a.shape)
        i = 0
        frame = 0
        while i < len(a) and frame < len(pitch):
            period = int(1.0 / pitch[frame] * r)
            #period = int(1.0 / 200 * r)
            ew[i] = np.sqrt(hnr[frame] / (hnr[frame] + 1)) * period
            i += period
            frame = i // framePeriod
        e = Frame(ew, size=frameSize, period=framePeriod)
        pitchCorrection = 1
        for i in range(len(e)):
            e[i] += np.random.normal(size=frameSize) * \
                    np.sqrt(1.0 / (hnr[i] + 1)) / pitchCorrection
        e *= pitchCorrection
        
    e = Window(e, hw)
    s = ARResynthesis(e, ar, g)
    print "wav: ", saveFile
    if ola:
        #s = Window(s, hw)
        s = OverlapAdd(s)
        WavSink(s, saveFile, r)
    else:
        WavSink(s.flatten('A'), saveFile, r)

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
