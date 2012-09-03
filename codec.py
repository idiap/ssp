#!/usr/bin/env python2
#
# Copyright 2012 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, May 2012
#
from ssp import *
from optparse import OptionParser
from os.path import splitext

# Command line
op = OptionParser(usage="usage: %prog [options] [inFile outFile]")
op.add_option("-r", dest="rate", default='16000',
              help="Sample rate")
op.add_option("-f", dest="fileList",
              help="List of input output file pairs")
op.add_option("-e", dest="encode", action="store_true", default=False,
              help="Encode source files")
op.add_option("-d", dest="decode", action="store_true", default=False,
              help="Decode source files")
op.add_option("-o", dest="ola", action="store_false", default=True,
              help="Concatenate frames (i.e., don't use OLA)")
op.add_option("-x", dest="excitation", action="store_true", default=False,
              help="Output excitation waveform instead of encoding")
(opt, arg) = op.parse_args()

# For excitation we need to disable OLA
if opt.excitation:
    opt.ola = False

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

pz = False

# Sample rate specific default parameters
framePeriod = {
    8000: 128,
    16000: 128,
    22050: 256
}

lpOrder = {
    8000: 10,
    16000: 24,
    22050: 24
}

def encode(a):
    """
    Encode a speech waveform.  The encoding framers (frames and pitch)
    pad the frames so that the first frame is centered on sample zero.
    This is consistent with STRAIGHT and SPTK (I hope!).  At least, it
    means the pitch can have longer frame lengths and still align with
    the OLA'd frames.
    """
    if opt.ola:
        frameSize = framePeriod[r] * 2
    else:
        frameSize = framePeriod[r]

    # First the pitch as it's on the unaltered waveform.  The frame
    # should be long with no window.
    pitchSize = 1024
    pf = Frame(a, size=pitchSize, period=framePeriod[r])
    pitch, hnr = ACPitch(pf)

    # Now mess with with pre-emphasis and the like for the LP
    if pz:
        a = ZeroMean(np.array(a))
        a = PoleFilter(a, 1.0)
    f = Frame(a, size=frameSize, period=framePeriod[r])
    aw = np.hanning(frameSize+1)
    aw = np.delete(aw, -1)
    w = Window(f, aw)
    ac = Autocorrelation(w)
    lp = Parameter('AR', 'levinson')
    if lp == 'levinson':
        ar, g = ARLevinson(ac, lpOrder[r])
    elif lp == 'ridge':
        ar, g = ARRidge(ac, lpOrder[r], 0.03)
    elif lp == 'lasso':
        ar, g = ARLasso(ac, lpOrder[r], 10)
    elif lp == 'sparse':
        ar, g = ARSparse(f, lpOrder[r])

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

    if opt.excitation:
        e = ARExcitation(f, ar, g)
        return e.flatten('C')/frameSize

    return (ar, g, pitch, hnr)


def decode((ar, g, pitch, hnr)):
    """
    Decode a speech waveform.
    """
    nFrames = len(ar)
    assert(len(g) == nFrames)
    assert(len(pitch) == nFrames)
    assert(len(hnr) == nFrames)

    # The original framer padded the ends so the number of samples to
    # syntheisise is a bit less than you might think
    if opt.ola:
        frameSize = framePeriod[r] * 2
        nSamples = framePeriod[r] * (nFrames-1)
    else:
        frameSize = framePeriod[r]
        nSamples = frameSize * (nFrames-1)

    ex = Parameter('Excitation', 'synth')
    if ex == 'ar':
        e = ARExcitation(f, ar, g)
    elif ex == 'noise':
        e = np.random.normal(size=f.shape)
    elif ex == 'robot':
        ew = np.zeros(nSamples)
        period = int(1.0 / 200 * r)
        for i in range(0, len(ew), period):
            ew[i] = period
        e = Frame(ew, size=frameSize, period=framePeriod[r])        
    elif ex == 'synth':
        # Harmonic part
        mperiod = int(1.0 / np.mean(pitch) * r)
        ptype = Parameter('Pulse', 'impulse')
        pr, pg = pulse_response(ptype, period=mperiod, order=lpOrder[r])
        h = np.zeros(nSamples)
        i = 0
        frame = 0
        while i < nSamples and frame < len(pitch):
            period = int(1.0 / pitch[frame] * r)
            if i + period > nSamples:
                break
            weight = np.sqrt(hnr[frame] / (hnr[frame] + 1))
            h[i:i+period] = pulse(period, ptype) * weight
            i += period
            frame = i // framePeriod[r]
        h = ARExcitation(h, pr, 1.0)
        fh = Frame(h, size=frameSize, period=framePeriod[r])

        # Noise part
        n = np.random.normal(size=nSamples)
        n = ZeroFilter(n, 1.0) # Include the radiation impedance
        fn = Frame(n, size=frameSize, period=framePeriod[r])
        for i in range(len(fn)):
            fn[i] *= np.sqrt(1.0 / (hnr[i] + 1))
        e = fn + fh
    elif ex == 'sine':
        order = 20
        sine = Harmonics(r, order)
        h = np.zeros(nSamples)
        for i in range(0, len(h)-framePeriod[r], framePeriod[r]):
            frame = i // framePeriod[r]
            period = int(1.0 / pitch[frame] * r)
            weight = np.sqrt(hnr[frame] / (hnr[frame] + 1))
            h[i:i+framePeriod[r]] = sine.sample(pitch[frame], framePeriod[r]) * weight
        fh = Frame(h, size=frameSize, period=framePeriod[r])
        n = np.random.normal(size=nSamples)
        fn = Frame(n, size=frameSize, period=framePeriod[r])
        for i in range(len(fn)):
            fn[i] *= np.sqrt(1.0 / (hnr[i] + 1))
        e = fn + fh*10
    elif ex == 'holp':
        # Some noise
        n = np.random.normal(size=nSamples)
        fn = Frame(n, size=frameSize, period=framePeriod[r])

        # Use the noise to excite a high order AR model
        fh = np.ndarray(fn.shape)
        for i in range(len(fn)):
            hoar = ARHarmonicPoly(pitch[i], r, 0.7)
            fh[i] = ARResynthesis(fn[i], hoar, 1.0 / linalg.norm(hoar)**2)
            print i, pitch[i], linalg.norm(hoar), np.min(fh[i]), np.max(fh[i])
            print ' ', np.min(hoar), np.max(hoar)
            # fh[i] *= np.sqrt(r / pitch[i]) / linalg.norm(fh[i])
            # fh[i] *= np.sqrt(hnr[i] / (hnr[i] + 1))

        # Weight the noise as for the other methods
        for i in range(len(fn)):
            fn[i] *= np.sqrt(1.0 / (hnr[i] + 1))
        e = fh # fn + fh*30
        
    else:
        exit

    # Asymmetric window for OLA
    sw = np.hanning(frameSize+1)
    sw = np.delete(sw, -1)

    #e = Window(e, sw)
    s = ARResynthesis(e, ar, g)
    if opt.ola:
        s = Window(s, sw)
        s = OverlapAdd(s)
        if pz:
            s = ZeroFilter(s, 1.0)
    return s.flatten('C')

#
# Main loop over the file list
#
r = int(opt.rate)
for pair in pairs:
    loadFile, saveFile = pair.strip().split()

    # Neither flag - assume a best effort copy
    if not (opt.encode or opt.decode):
        rate, a = WavSource(loadFile)
        if rate != r:
            raise ValueError("Wrong sample rate")
        if opt.excitation:
            e = encode(a)
            WavSink(e, saveFile, r)
        else:
            d = decode(encode(a))
            WavSink(d, saveFile, r)

    # Encode to a file
    if opt.encode:
        rate, a = WavSource(loadFile)
        if rate != r:
            raise ValueError("Wrong sample rate")
        (ar, g, pitch, hnr) = encode(a)

        # The cepstrum part is just like HTK
        c = ARCepstrum(ar, g, lpOrder[r])
        period = float(framePeriod[r])/r
        HTKSink(saveFile, c, period)

        # F0 and HNR are both text formats
        (path, ext) = splitext(saveFile)
        saveFileLF0 = path + ".f0"
        saveFileHNR = path + ".hnr"
        np.savetxt(saveFileLF0, np.log(pitch))
        np.savetxt(saveFileHNR, hnr)

    # Decode from a file
    if opt.decode:
        (path, ext) = splitext(loadFile)
        loadFileLF0 = path + ".f0"
        loadFileHNR = path + ".hnr"
        pitch = np.exp(np.loadtxt(loadFileLF0))
        hnr = np.loadtxt(loadFileHNR)
        c, period = HTKSource(loadFile)
        (ar, g) = ARCepstrumToPoly(c)
        d = decode((ar, g, pitch, hnr))
        WavSink(d, saveFile, r)
