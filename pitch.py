#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#
from optparse import OptionParser
op = OptionParser()
(option, arg) = op.parse_args()
if (len(arg) < 1):
    print "Need one arg"
    exit(1)
file = arg[0]


def kalman(obs, obsVar, seqVar, initMean, initVar):
    stateMean = np.ndarray(len(obs))
    stateVar  = np.ndarray(len(obs))

    # Initialise
    stateMean[0] = ( (obs[0] * initVar + initMean * obsVar[0]) /
                     (initVar + obsVar[0]) )
    stateVar[0]  = initVar * obsVar[0] / (initVar + obsVar[0])

    # Filter loop
    for i in range(1, len(obs)):
        predictor = seqVar + stateVar[i-1]
        stateMean[i] = ( (obs[i] * predictor + stateMean[i-1] * obsVar[i]) /
                         (predictor + obsVar[i]) )
        stateVar[i]  = predictor * obsVar[i] / (predictor + obsVar[i])

    # Smoother loop
    for i in reversed(range(len(obs)-1)):
        stateMean[i] = ( stateMean[i+1] * stateVar[i] +
                         stateMean[i]   * seqVar )
        stateMean[i] /= (seqVar + stateVar[i])
        J = stateVar[i] / (stateVar[i] + seqVar)
        stateVar[i] = J * (seqVar + J * stateVar[i+1])

    return stateMean, stateVar

from ssp import *
import numpy as np
import matplotlib.pyplot as plt

fs = 512
fp = 256

loPitch = 40
hiPitch = 1000

loPeriod = 1.0 / loPitch
hiPeriod = 1.0 / hiPitch

# Load and process
r, a = WavSource(file)

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

# Basic spectral analysis, windowed, for reference
w = np.hanning(fs)
#w = gaussian(fs)
a = ZeroFilter(a)
f = Frame(a, size=fs, period=fp)
wf = Window(f, w)
p = Periodogram(wf)

# Plot
fig = Figure(4,1)
pSpec = fig.subplot()
specplot(pSpec, p[:,:p.shape[1]/2+1], r)


method = Parameter('Method', 'ac')

if method == 'ac':
    # Autocorrelation method, loosely after Boersma
    ac = Autocorrelation(p, 'psd')
    for i in range(len(ac)):
        ac[i] /= ac[i, 0]
    wac = Autocorrelation(w)
    wac /= wac[0]
    nac = Divide(ac, wac)

    fPlot = fig.subplot()
    fPlot.set_xlim(0, fs)
    frame = Parameter("Frame", 10)
    fPlot.plot(nac[frame], 'c')

    m = np.argmax(nac[:,loACBin:hiACBin], axis=1) + loACBin
    pitch = np.ndarray(len(m))
    hnr = np.ndarray(len(m))
    var = np.ndarray(len(m))
    for i in range(len(m)):
        pitch[i] = 1.0 / acbin_to_seconds(m[i], r)
        hnr[i] = nac[i, m[i]] / (1.0 - nac[i, m[i]])
        # var[i] = -np.log(nac[i, m[i]])
        var[i] = (1.0 / hnr[i])**2 * hiPitch

    pPlot = fig.subplot()
    pPlot.set_xlim(0, len(pitch))
    pPlot.plot(pitch, 'r')
    #pPlot.plot(hnr * hiPitch, 'g')
    #pPlot.plot(var * hiPitch, 'b')

    # Kalman smoother
    kpitch, kvar = kalman(pitch, var, 10.0, hiPitch+loPitch/2, hiPitch*hiPitch)
    pPlot.plot(kpitch, 'c')

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
