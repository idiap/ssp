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

# Load and process
print "Using file:", file
r, a = WavSource(file)

fs = 1024
fp = 256

# Basic spectral analysis
a = ZeroFilter(a)
f = Frame(a, size=fs, period=fp)
w = Window(f, nuttall(fs))
p = Periodogram(w)

# Plot
fig = plt.figure()


method = Parameter('method', 'map')

if method == 'ar':
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
    c = ARPoly(a)
    m, s = ARAngle(c)

elif method == 'map':
    print "not done yet"


if 1:
    pSpec = fig.add_subplot(4,1,1)
    epSpec = fig.add_subplot(4,1,2)
    lSpec = fig.add_subplot(4,1,3)
    rSpec = fig.add_subplot(4,1,4)

    specplot(pSpec, p[:,:p.shape[1]/2+1], r)
    specplot(epSpec, ep[:,:ep.shape[1]/2+1], r)
    specplot(lSpec, l, r)

    rSpec.set_xlim(0, len(m)-1)
    rSpec.plot(m / np.pi * r, 'r')
    rSpec.plot((m+s) / np.pi * r, 'b')
    rSpec.plot((m-s) / np.pi * r, 'b')
else:
    f = Parameter("Frame", 10)
    zplot(fig, c[f])

plt.show()
