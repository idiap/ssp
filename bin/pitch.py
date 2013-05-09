#!/usr/bin/env python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#

#
# Notes:
#
# You'd think that log(f0) would be the thing to measure.  It doesn't
# really work in the Kalman smoother though.  The mean gets pulled up
# or down, rather than remaining steady when there is high variance.
#
# Autocorrelation on the excitation is very noisy for 16 kHz, but OK
# for 8 kHz.  Ergo, the harmonic excitation itself is band-limited.
#

from optparse import OptionParser
op = OptionParser()
(option, arg) = op.parse_args()
if (len(arg) < 1):
    print "Need one arg"
    exit(1)
wavFile = arg[0]

from ssp import *

# Reference f0 from tempo
tempo = None
if len(arg) > 1:
    tempo = np.loadtxt(arg[1])

# Load and process
pcm = PulseCodeModulation()
a = pcm.WavSource(wavFile)
#r, a = wavfile.read(wavFile)

# Default to 8k
fs = 512
fp = 256
if pcm.rate == 16000:
    fs = 1024
    fp = 128
if pcm.rate == 22050:
    fs = 2048
    fp = 256
elif pcm.rate == 96000:
    fs = 8192
    fp = 160

print "Frame period", fp, "is", fp/float(pcm.rate)*1000, "ms, or", float(pcm.rate) / fp, "Hz"

loPitch = 40
hiPitch = 500

loPeriod = 1.0 / loPitch
hiPeriod = 1.0 / hiPitch

loDFTBin = pcm.hertz_to_dftbin(loPitch, fs)
hiDFTBin = pcm.hertz_to_dftbin(hiPitch, fs)
print "Pitch range is bins", loDFTBin, "to", hiDFTBin

loACBin = pcm.seconds_to_acbin(hiPeriod)
hiACBin = pcm.seconds_to_acbin(loPeriod)
print "Period range is bins", loACBin, "to", hiACBin

# The AC bin for the period of the lowest frequency needs to be
# smaller than the size of the AC.
if hiACBin >= fs / 2:
    print "Frame size {0} too small for pitch {1} Hz".format(fs, loPitch)

# Add noise
#a += np.random.normal(size=len(a)) * 1e-3

# Basic spectral analysis, windowed, for reference.  Don't do
# pre-emphasis; it will break low F0 speakers.
#w = np.hanning(fs)
w = gaussian(fs)
f = Frame(a, size=fs, period=fp)
f = ZeroMean(f)
wf = Window(f, w)
p = Periodogram(wf)

# Plot
fig = Figure(5,1)
#pSpec = fig.subplot()
#specplot(pSpec, p, r)
if True:
    sSpec = fig.subplot()
    specplot(sSpec, p[:,:pcm.hertz_to_dftbin(hiPitch, fs)], hiPitch*2)

method = parameter('Method', 'ac')

if method == 'ac':
    if False:
        # Low order AR
        order = 3
        a = Autocorrelation(p, 'psd')
        la, lg = ARRidge(a, order, ridge=0.8)
        f = ARExcitation(wf, la, lg)
        f = ZeroMean(f)
        p = Periodogram(f)

    # Autocorrelation method, loosely after Boersma
    ac = Autocorrelation(p, 2, 'psd')
    for i in range(len(ac)):
        ac[i] /= ac[i, 0]
    wac = Autocorrelation(w)
    wac /= wac[0]
    nac = Divide(ac, wac)
    #h = Harmonogram(nac, 'psd', True)

    if True:
        fPlot = fig.subplot()
        frame = ssp.parameter("Frame", 0)
        #fPlot.plot(f[frame], 'r')
        #fPlot.plot(wf[frame], 'g')
        fPlot.plot(ac[frame], 'b')
        fPlot.plot(nac[frame], 'c')
        #fPlot.plot(h[frame], 'r')
        fPlot.set_xlim(0, fs/2)

    # Pitch bin is the maximum in each frame
    #m = np.argmax(nac[:,loACBin:hiACBin], axis=1) + loACBin
    m = Argmax(nac, loACBin, hiACBin)

    # Convert to pitch and harmonic noise ratio
    pitch = np.ndarray(len(m))
    hnr = np.ndarray(len(m))
    var = np.ndarray(len(m))
    prange = hiPitch - loPitch
    for i in range(len(m)):
        pitch[i] = 1.0 / pcm.acbin_to_seconds(m[i])
        fnac = np.max([nac[i, m[i]], 1e-6])
        if (nac[i, m[i]-1] > nac[i, m[i]]) or (nac[i, m[i]+1] > nac[i, m[i]]):
            # No peak found; set HNR small
            hnr[i] = 1e-8
        else:
            hnr[i] = fnac / (1.0 - fnac)
        var[i] = (1.0 / hnr[i] * prange)**2

    stddev = np.sqrt(var)
    if True:
        hPlot = fig.subplot()
        hPlot.plot(pitch, 'c')
        hPlot.plot(pitch + stddev, 'b')
        hPlot.plot(pitch - stddev, 'b')
        hPlot.set_xlim(0, len(pitch))
        hPlot.set_ylim(0, hiPitch)

    # Kalman smoother
    kPitch, kVar = kalman(
        pitch, var, 1e3, loPitch + prange/2, prange**2
        )
    stddev = np.sqrt(kVar)

    if True:
        pPlot = fig.subplot()
        pPlot.plot(pitch, 'r')
        pPlot.plot(kPitch, 'c')
        pPlot.plot(kPitch + stddev, 'b')
        pPlot.plot(kPitch - stddev, 'b')
        pPlot.set_xlim(0, len(pitch))
        pPlot.set_ylim(0, hiPitch)

    # Now run it again, but with tighter limits
    mpitch = np.mean(kPitch)
    for i in range(len(nac)):
        hi = pcm.seconds_to_acbin(1.0 / (kPitch[i] * 0.75))
        lo = pcm.seconds_to_acbin(1.0 / (kPitch[i] * 1.5))
        rng = hi - lo
        loBin = np.max([lo, loACBin])
        hiBin = np.min([hi, hiACBin])
        m[i] = np.argmax(nac[i, loBin:hiBin]) + loBin
        pitch[i] = 1.0 / pcm.acbin_to_seconds(m[i])
        fnac = np.max([nac[i, m[i]], 1e-6])
        if (nac[i, m[i]-1] > nac[i, m[i]]) or (nac[i, m[i]+1] > nac[i, m[i]]):
            # No peak found; set HNR small
            hnr[i] = 1e-8
        else:
            hnr[i] = fnac / (1.0 - fnac)
        var[i] = (1.0 / hnr[i])**2 * rng**2
    
    # Kalman smoother again
    kPitch, kVar = kalman(
        pitch, var, 1e4, mpitch, prange**2
        )
    stddev = np.sqrt(kVar)

    if True:
        sPlot = fig.subplot()
        sPlot.plot(pitch, 'r')
        sPlot.plot(tempo, 'm')
        sPlot.plot(kPitch, 'c')
        sPlot.plot(kPitch + stddev, 'b')
        sPlot.plot(kPitch - stddev, 'b')
        sPlot.set_xlim(0, len(pitch))
        sPlot.set_ylim(0, hiPitch)

elif method == 'ar':
    # Low order AR
    order = 15
    a = Autocorrelation(wf)
    la, lg = ARLevinson(a, order)
    f = ARExcitation(f, la, lg)

    # High order AR
    order = 150
    fh = Window(f, np.hanning(fs))
    ep = Periodogram(fh)
    a = Autocorrelation(fh)
    a, g = ARLasso(a, order, 500)
    l = ARSpectrum(a, g, nSpec=fs/2)

    epSpec = fig.subplot()
    lSpec = fig.subplot()
    specplot(epSpec, ep[:,:ep.shape[1]/2+1], pcm.rate)
    specplot(lSpec, l, pcm.rate)

    c = ARPoly(a)
    m, s = ARAngle(c)

    if 1:
        rSpec = fig.subplot()
        rSpec.set_xlim(0, len(m)-1)
        rSpec.plot(m / np.pi * pcm.rate, 'r')
        rSpec.plot((m+s) / np.pi * pcm.rate, 'b')
        rSpec.plot((m-s) / np.pi * pcm.rate, 'b')
    else:
        f = Parameter("Frame", 10)
        zplot(fig, c[f])

elif method == 'map':
    h = Harmonogram(p, 'psd')
    hSpec = fig.subplot()
    specplot(hSpec, h, pcm.rate)

    # Low order AR
    order = 15
    a = Autocorrelation(f)
    la, lg = ARLevinson(a, order)
    e = ARExcitation(f, la, lg)

    eh = Harmonogram(e)
    ehSpec = fig.subplot()
    specplot(ehSpec, eh, pcm.rate)

    frame = Parameter('Frame', 1)
    rSpec = fig.subplot()
    rSpec.set_xlim(0, hiDFTBin-1)
    
    rSpec.plot(np.divide(p[frame,:hiDFTBin], Norm(p[frame,:hiDFTBin], 2)), 'c')
#    rSpec.plot(h[frame,:hiDFTBin] / Norm(h[frame,:hiDFTBin], 2), 'r')
#    rSpec.plot(eh[frame,:hiDFTBin] / Norm(eh[frame,:hiDFTBin], 2), 'b')

plt.show()
