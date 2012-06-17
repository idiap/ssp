#!/usr/bin/env python
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
r, a = WavSource(wavFile)

# Default to 8k
fs = 512
fp = 256
if r == 16000:
    fs = 1024
    fp = 160

loPitch = 40
hiPitch = 500

loPeriod = 1.0 / loPitch
hiPeriod = 1.0 / hiPitch

loDFTBin = hertz_to_dftbin(loPitch, fs, r)
hiDFTBin = hertz_to_dftbin(hiPitch, fs, r)
print "Pitch range is bins", loDFTBin, "to", hiDFTBin

loACBin = seconds_to_acbin(hiPeriod, r)
hiACBin = seconds_to_acbin(loPeriod, r)
print "Period range is bins", loACBin, "to", hiACBin

# The AC bin for the period of the lowest frequency needs to be
# smaller than the size of the AC.
if hiACBin >= fs / 2:
    print "Frame size {} too small for pitch {} Hz".format(fs, loPitch)

# Basic spectral analysis, windowed, for reference.  Don't do
# pre-emphasis; it will break low F0 speakers.
#w = np.hanning(fs)
w = gaussian(fs)
f = Frame(a, size=fs, period=fp)
f = ZeroMean(f)
wf = Window(f, w)
p = Periodogram(wf)

# Plot
fig = Figure(4,1)
pSpec = fig.subplot()
specplot(pSpec, p[:,:p.shape[1]/2+1], r)


method = Parameter('Method', 'ac')

if method == 'ac':
    if False:
        # Low order AR
        order = 10
        a = Autocorrelation(p, 'psd')
        la, lg = ARLevinson(a, order)
        f = ARExcitation(wf, la, lg)
        f = ZeroMean(f)
        p = Periodogram(f)

    # Autocorrelation method, loosely after Boersma
    ac = Autocorrelation(p, 'psd')
    for i in range(len(ac)):
        ac[i] /= ac[i, 0]
    wac = Autocorrelation(w)
    wac /= wac[0]
    nac = Divide(ac, wac)

    if False:
        fPlot = fig.subplot()
        fPlot.set_xlim(0, fs)
        frame = Parameter("Frame", 10)
        #fPlot.plot(f[frame], 'r')
        fPlot.plot(wf[frame], 'g')
        #fPlot.plot(ac[frame], 'b')
        #fPlot.plot(nac[frame], 'c')

    # Pitch bin is the maximum in each frame
    m = np.argmax(nac[:,loACBin:hiACBin], axis=1) + loACBin

    # Convert to pitch and harmonic noise ratio
    pitch = np.ndarray(len(m))
    hnr = np.ndarray(len(m))
    var = np.ndarray(len(m))
    prange = hiPitch - loPitch
    for i in range(len(m)):
        pitch[i] = 1.0 / acbin_to_seconds(m[i], r)
        hnr[i] = nac[i, m[i]] / (1.0 - nac[i, m[i]])
        var[i] = (1.0 / hnr[i])**2 * prange**2

    if False:
        hPlot = fig.subplot()
        hPlot.plot(hnr)
        hPlot.plot(1/hnr)
        hPlot.plot(1/hnr**2)
        hPlot.set_xlim(0, len(pitch))
        hPlot.set_ylim(0, 5)

    sSpec = fig.subplot()
    specplot(sSpec, p[:,:hertz_to_dftbin(hiPitch, fs, r)], hiPitch*2)

    pPlot = fig.subplot()
    pPlot.plot(pitch, 'r')

    # Kalman smoother
    kPitch, kVar = kalman(
        pitch, var, 1e4, loPitch + prange/2, prange**2
        )
    stddev = np.sqrt(kVar)
    pPlot.plot(kPitch, 'c')
    pPlot.plot(kPitch + stddev, 'b')
    pPlot.plot(kPitch - stddev, 'b')
    pPlot.set_xlim(0, len(pitch))
    pPlot.set_ylim(0, hiPitch)

    # Now run it again, but with tighter limits
    mpitch = np.mean(kPitch)
    for i in range(len(nac)):
        hi = seconds_to_acbin(1.0 / (kPitch[i] * 0.75), r)
        lo = seconds_to_acbin(1.0 / (kPitch[i] * 1.5), r)
        rng = hi - lo
        loBin = np.max([lo, loACBin])
        hiBin = np.min([hi, hiACBin])
        m[i] = np.argmax(nac[i, loBin:hiBin]) + loBin
        pitch[i] = 1.0 / acbin_to_seconds(m[i], r)
        hnr[i] = nac[i, m[i]] / (1.0 - nac[i, m[i]])
        var[i] = (1.0 / hnr[i])**2 * rng**2
    
    sPlot = fig.subplot()
    sPlot.plot(pitch, 'r')
    sPlot.plot(tempo, 'm')

    # Kalman smoother again
    kPitch, kVar = kalman(
        pitch, var, 1e8, mpitch, prange**2
        )
    stddev = np.sqrt(kVar)
    sPlot.plot(kPitch, 'c')
    sPlot.plot(kPitch + stddev, 'b')
    sPlot.plot(kPitch - stddev, 'b')
    sPlot.set_xlim(0, len(pitch))
    sPlot.set_ylim(0, hiPitch)

elif method == 'ar':
    # Low order AR
    order = 15
    a = Autocorrelation(w)
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
    specplot(epSpec, ep[:,:ep.shape[1]/2+1], r)
    specplot(lSpec, l, r)

    c = ARPoly(a)
    m, s = ARAngle(c)

    if 1:
        rSpec = fig.subplot()
        rSpec.set_xlim(0, len(m)-1)
        rSpec.plot(m / np.pi * r, 'r')
        rSpec.plot((m+s) / np.pi * r, 'b')
        rSpec.plot((m-s) / np.pi * r, 'b')
    else:
        f = Parameter("Frame", 10)
        zplot(fig, c[f])

elif method == 'map':
    h = Harmonogram(p, 'psd')
    hSpec = fig.subplot()
    specplot(hSpec, h, r)

    # Low order AR
    order = 15
    a = Autocorrelation(f)
    la, lg = ARLevinson(a, order)
    e = ARExcitation(f, la, lg)

    eh = Harmonogram(e)
    ehSpec = fig.subplot()
    specplot(ehSpec, eh, r)

    frame = Parameter('Frame', 1)
    rSpec = fig.subplot()
    rSpec.set_xlim(0, hiDFTBin-1)
    
    rSpec.plot(np.divide(p[frame,:hiDFTBin], Norm(p[frame,:hiDFTBin], 2)), 'c')
#    rSpec.plot(h[frame,:hiDFTBin] / Norm(h[frame,:hiDFTBin], 2), 'r')
#    rSpec.plot(eh[frame,:hiDFTBin] / Norm(eh[frame,:hiDFTBin], 2), 'b')

plt.show()
