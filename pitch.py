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

from ssp import *
import numpy as np
import matplotlib.pyplot as plt

fs = 256
fp = 256

loPitch = 40
hiPitch = 1000

# Load and process
r, a = WavSource(file)

loBin = hertz_to_bin(loPitch, fs, r)
hiBin = hertz_to_bin(hiPitch, fs, r)
print "Pitch range is", loBin, "to", hiBin

# Basic spectral analysis
a = ZeroFilter(a)
f = Frame(a, size=fs, period=fp)
#w = Window(f, nuttall(fs))
p = Periodogram(f)

# Plot
fig = Figure(4,1)
pSpec = fig.subplot()
specplot(pSpec, p[:,:p.shape[1]/2+1], r)


method = Parameter('Method', 'map')

if method == 'ac':
    ac = Autocorrelation(p, 'psd')
    acSpec = fig.subplot()
    specplot(acSpec, ac, r)

    fPlot = fig.subplot()
    fPlot.set_xlim(0, fs)
    frame = Parameter("Frame", 10)
    fPlot.plot(np.divide(ac[frame], Norm(p[frame], 2)), 'c')

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
    rSpec.set_xlim(0, hiBin-1)
    
    rSpec.plot(np.divide(p[frame,:hiBin], Norm(p[frame,:hiBin], 2)), 'c')
#    rSpec.plot(h[frame,:hiBin] / Norm(h[frame,:hiBin], 2), 'r')
#    rSpec.plot(eh[frame,:hiBin] / Norm(eh[frame,:hiBin], 2), 'b')

plt.show()
