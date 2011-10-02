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

a = ZeroFilter(a)
f = Frame(a, size=fs, period=fp)

# Low order AR
order = 15
fn = Window(f, nuttall(fs))
op = Periodogram(fn)
a = Autocorrelation(fn)
la, lg = ARLevinson(a, order)


f = ARExcitation(f, la)
fh = Window(f, np.hanning(fs))
ep = Periodogram(fh)
a = Autocorrelation(fh)


# High order AR
order = 70
a, g = ARLasso(a, order, 100)

l = ARSpectrum(a, g, nSpec=fs/2)

def ARPoly(a):
    if a.ndim > 1:
        ret = np.ndarray(a.shape, dtype='complex')
        for f in range(a.shape[0]):
            ret[f] = ARPoly(a[f])
        return ret

    # The refection coeffs are negative, so insert -1 and assume that
    # the whole thing can be multiplied by -1
    r = np.roots(np.insert(a, 0, -1))
    return r

def Angle(a):
    if a.ndim > 1:
        ret_m = np.ndarray(a.shape)
        ret_s = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret_m[f], ret_s[f] = Angle(a[f])
        return ret_m, ret_s

    # First extract the angles of large poles above the real line
    t = np.zeros(len(a))
    j = 0
    for i in range(len(a)):
        if np.abs(a[i]) > 0.8 and np.imag(a[i]) > 0:
            t[j] = np.angle(a[i])
            j += 1

    # Build an array of the differences between the sorted angles
    t = np.sort(t[:j])
    for i in range(1,j):
        t[i-1] = t[i] - t[i-1]

    # We need the mean and stddev of those differences
    m = np.mean(t[:j-1])
    s = np.std(t[:j-1])
    return m, s

c = ARPoly(a)
m, s = Angle(c)

# Draw it
# fig.add_subplot(2,1,1) # two rows, one column, first plot
fig = plt.figure()

if 1:
    opSpec = fig.add_subplot(4,1,1)
    epSpec = fig.add_subplot(4,1,2)
    lSpec = fig.add_subplot(4,1,3)
    rSpec = fig.add_subplot(4,1,4)

    specplot(opSpec, op[:,:op.shape[1]/2+1], r)
    specplot(epSpec, ep[:,:ep.shape[1]/2+1], r)
    specplot(lSpec, l, r)

    rSpec.set_xlim(0, len(m)-1)
    rSpec.plot(m / np.pi * r, 'r')
    rSpec.plot((m+s) / np.pi * r, 'b')
    rSpec.plot((m-s) / np.pi * r, 'b')
else:
    zplot(fig, c[77])

plt.show()
