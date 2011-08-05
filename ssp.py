#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, August 2011
#
import numpy as np
import numpy.linalg as linalg
from scipy.io import wavfile

def WavSource(file):
    rate, audio = wavfile.read(file)
    if audio.dtype == 'int16':
        audio = np.cast['float'](audio)
        audio /= 32768
    return rate, audio

def ZeroFilter(a):
    zero = 0.97
    filter = np.zeros(a.size)
    store = a[0]
    for i in range(a.size):
        filter[i] = a[i] - zero * store
        store = a[i]
    return filter

def Frame(a, size=512, period=256):
    nFrames = (a.size - (size-period)) // period
    frame = np.zeros((nFrames, size))
    for r in range(nFrames):
        s = r*period
        e = s+size
        frame[r, :] = a[s:e]
    return frame

def Energy(a):
    e = np.zeros(a.shape[0])
    for r in range(a.shape[0]):
        e[r] = linalg.norm(a[r,:])**2
    return e

def Periodogram(a):
    psd = np.zeros(a.shape)
    for r in range(a.shape[0]):
        dft = np.fft.fft(a[r, :])
        psd[r, :] = np.abs(dft)**2
    return psd

def Autocorrelation(a):
    ac = np.zeros(a.shape)
    for r in range(a.shape[0]):
        psd = abs(np.fft.fft(a[r, :]))**2
        dft = np.fft.ifft(psd)
        ac[r, :] = np.real(dft)
    return ac

def ARMatrix(a, order=10, method='matrix'):
    coeff = np.zeros((a.shape[0], order))
    gain = np.ones(a.shape[0])
    if method == 'matrix':
        for r in range(a.shape[0]):
            Y = Frame(a[r,:a.shape[1]-1], size=order, period=1)
            y = a[r,order:]
            YY = np.dot(Y.T,Y)
            Yy = np.dot(Y.T,y)
            elop = np.dot(linalg.inv(YY), Yy)

            for i in range(order):
                coeff[r,i] = elop[order-i-1]
            gain[r] = (np.dot(y,y) - np.dot(elop,Yy)) / y.size
    elif method == 'acmatrix':
        ac = Autocorrelation(a)
        YY = np.zeros((order, order))
        Yy = np.zeros(order)
        for r in range(a.shape[0]):
            for i in range(order):
                Yy[i] = ac[r,i+1]
                for j in range(order):
                    YY[i,j] = ac[r,abs(i-j)]
            coeff[r,:] = np.dot(linalg.inv(YY), Yy)
            gain[r] = (ac[r,0] - np.dot(coeff[r,:],Yy)) / a.shape[1]
    else:
        print "Unknown AR method"
        exit(1)

    return (coeff, gain)

def ARLevinson(ac, order=10):
    tmp0 = np.zeros(order)
    tmp1 = np.zeros(order)
    curr = tmp0.view()
    prev = tmp1.view()
    coeff = np.zeros((ac.shape[0], order))
    gain = np.ones(ac.shape[0])
    for r in range(ac.shape[0]):
        curr.fill(0)
        prev.fill(0)
        error = ac[r,0]
        if error < 1e-8:
            print "error: ", error
        for i in range(order):
            # swap current and previous coefficients
            tmp = curr.view()
            curr = prev.view()
            prev = tmp.view()

            # Recurse
            k = ac[r,i+1]
            for j in range(i):
                k -= prev[j] * ac[r, i-j]
            curr[i] = k / error
            error *= 1 - curr[i]**2
            for j in range(i):
                curr[j] = prev[j] - curr[i] * prev[i-j-1]

        # Whichever curr is viewing is the output
        coeff[r,:] = curr

        # Dot product would be better
        gain[r] = ac[r, 0]
        for i in range(order):
            gain[r] -= coeff[r,i] * ac[r,i+1]

    gain /= ac.shape[1]
    return (coeff, gain)

# Compute power spectrum
def ARSpectrum(a, g, nSpec=256):
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], nSpec))
        for f in range(a.shape[0]):
            ret[f] = ARSpectrum(a[f], g[f], nSpec)
        return ret
    
    spec = np.ndarray(nSpec)
    for i in range(nSpec):
        omega = np.pi * i / nSpec
        sm = 0j
        for j in range(a.size):
            sm += a[j] * np.exp(-1j * omega * (j+1))
        spec[i] = g / abs(1 - sm)**2
    return spec

# Bilinear transform
def BilinearWarp(a, alpha=0.0, size=None):
    if a.ndim > 1:
        ret = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret[f] = BilinearWarp(a[f], alpha, size)
        return ret
    
    isize = a.size
    if size is None:
        osize = isize
    else:
        osize = size
    tmp0 = np.ndarray(osize)
    tmp1 = np.ndarray(osize)
    t = tmp0
    y = tmp1
    out = np.ndarray(osize)
    alpha2 = 1-alpha**2
    for r in range(a.shape[0]):
        t.fill(0)
        y.fill(0)
        for k in range(isize-1,-1,-1):
            t[0] = alpha * y[0] + a[k]
            t[1] = alpha * y[1] + alpha2 * y[0]
            for n in range(2,osize):
                t[n] = y[n-1] + alpha * (y[n] - t[n-1])

            tmp = y
            y = t
            t = tmp
        out = y

    return out

# In the AR case, we need to prepend a 1
def BilinearWarpAR(a, g, alpha=0):
    if a.ndim > 1:
        reta = np.ndarray(a.shape)
        retg = np.ndarray(g.shape)
        for f in range(a.shape[0]):
            reta[f], retg[f] = BilinearWarpAR(a[f], g[f], alpha)
        return reta, retg

    a = BilinearWarp(np.insert(a, 0, 1), alpha)
    g /= a[0]
    a /= a[0]
    return a[1:], g
