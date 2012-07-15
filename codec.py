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
import scipy.signal as ss

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
pz = False

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
        lpOrder = 24
    else:
        exit

    if ola:
        synthSize = framePeriod * 2
    else:
        frameSize = framePeriod
        synthSize = framePeriod
        
    # First the pitch as it's on the unaltered waveform.  The window
    # should be long with no sidelobes.
    pf = Frame(a, size=pitchSize, period=framePeriod)
    pitch, hnr = ACPitch(pf)

    # Now mess with with pre-emphasis and the like for the LP
    if pz:
        a = ZeroMean(np.array(a))
        a = PoleFilter(a, 1.0)
    f = Frame(a, size=frameSize, period=framePeriod)
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

    ex = Parameter('Excitation', 'synth')
    if ex == 'ar':
        e = ARExcitation(f, ar, g)
    elif ex == 'noise':
        e = np.random.normal(size=f.shape)
    elif ex == 'robot':
        ew = np.zeros(len(a))
        period = int(1.0 / 200 * r)
        for i in range(0, len(ew), period):
            ew[i] = period
        e = Frame(ew, size=synthSize, period=framePeriod)        
    elif ex == 'synth':
        # Harmonic part
        mperiod = int(1.0 / np.mean(pitch) * r)
        ptype = Parameter('Pulse', 'impulse')
        pr, pg = pulse_response(ptype, period=mperiod, order=lpOrder)
        h = np.zeros(len(a))
        i = 0
        frame = 0
        while i < len(a) and frame < len(pitch):
            period = int(1.0 / pitch[frame] * r)
            if i + period > len(a):
                break
            weight = np.sqrt(hnr[frame] / (hnr[frame] + 1))
            h[i:i+period] = pulse(period, ptype) * weight
            i += period
            frame = i // framePeriod
        h = ARExcitation(h, pr, 1.0)
        fh = Frame(h, size=synthSize, period=framePeriod)

        # Noise part
        n = np.random.normal(size=len(a))
        fn = Frame(ZeroFilter(n, 1.0), size=synthSize, period=framePeriod)
        for i in range(len(fn)):
            fn[i] *= np.sqrt(1.0 / (hnr[i] + 1))
        e = fn + fh*20
    elif ex == 'sine':
        order = 20
        sine = Harmonics(r, order)
        h = np.zeros(len(a))
        for i in range(0, len(h)-framePeriod, framePeriod):
            frame = i // framePeriod
            period = int(1.0 / pitch[frame] * r)
            weight = np.sqrt(hnr[frame] / (hnr[frame] + 1))
            h[i:i+framePeriod] = sine.sample(pitch[frame], framePeriod) * weight
        fh = Frame(h, size=synthSize, period=framePeriod)
        n = np.random.normal(size=len(a))
        fn = Frame(n, size=synthSize, period=framePeriod)
        for i in range(len(fn)):
            fn[i] *= np.sqrt(1.0 / (hnr[i] + 1))
        e = fn + fh*10
    else:
        exit
    sw = np.hanning(synthSize+1)
    sw = np.delete(sw, -1)

    #e = Window(e, sw)
    s = ARResynthesis(e, ar, g)
    print "wav: ", saveFile
    if ola:
        s = Window(s, sw)
        s = OverlapAdd(s)
        if pz:
            s = ZeroFilter(s, 1.0)
        WavSink(s.flatten('C'), saveFile, r)
    else:
        WavSink(e.flatten('C') / framePeriod, saveFile, r)

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
