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

# Load a .wav file
def WavSource(file):
    rate, audio = wavfile.read(file)
    if audio.dtype == 'int16':
        audio = np.cast['float'](audio)
        audio /= 32768
    return rate, audio

# Filter comprising a single zero
def ZeroFilter(a, zero=0.97):
    filter = np.zeros(a.size)
    store = a[0]
    for i in range(a.size):
        filter[i] = a[i] - zero * store
        store = a[i]
    return filter

# Convert array to framed array
def Frame(a, size=512, period=256):
    nFrames = (a.size - (size-period)) // period
    frame = np.zeros((nFrames, size))
    for r in range(nFrames):
        s = r*period
        e = s+size
        frame[r, :] = a[s:e]
    return frame

# Frame energy
def Energy(a):
    e = np.zeros(a.shape[0])
    for r in range(a.shape[0]):
        e[r] = linalg.norm(a[r,:])**2
    return e

# Lx norm
def Norm(a, L):
    e = np.zeros(a.shape[0])
    for r in range(a.shape[0]):
        e[r] = linalg.norm(a[r,:], ord=L)
    return e

def Periodogram(a):
    psd = np.ndarray(a.shape)
    for r in range(a.shape[0]):
        dft = np.fft.fft(a[r, :])
        psd[r, :] = np.abs(dft)**2
    return psd

def Autocorrelation(a):
    if a.ndim > 1:
        ret = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret[f] = Autocorrelation(a[f])
        return ret

    psd = abs(np.fft.fft(a))**2
    dft = np.fft.ifft(psd)
    ac = np.real(dft)/a.size
    return ac

def ARMatrix(a, order=10, method='matrix'):
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], order))
        gain = np.ndarray(a.shape[0])
        for f in range(a.shape[0]):
            ret[f], gain[f] = ARMatrix(a[f], order, method)
        return ret, gain

    coeff = np.zeros(order)
    if method == 'matrix':
        Y = Frame(a[:a.size-1], size=order, period=1)
        y = a[order:]
        YY = np.dot(Y.T,Y)
        Yy = np.dot(Y.T,y)
        elop = np.dot(linalg.inv(YY), Yy)
        for i in range(order):
            coeff[i] = elop[order-i-1]
        gain = (np.dot(y,y) - np.dot(elop,Yy)) / y.size
    elif method == 'acmatrix':
        ac = Autocorrelation(a)
        YY = np.ndarray((order, order))
        Yy = np.ndarray(order)
        for i in range(order):
            Yy[i] = ac[i+1] * a.size
            for j in range(order):
                YY[i,j] = ac[abs(i-j)] * a.size
        coeff = np.dot(linalg.inv(YY), Yy)
        gain = (ac[0] - np.dot(coeff,Yy / a.size))
    else:
        print "Unknown AR method"
        exit(1)

    return (coeff, gain)

# Raw Levinson-Durbin recursion
def levinson(ac, order, prior=0.0):
    curr = np.zeros(order)
    prev = np.zeros(order)
    error = ac[0] + prior
    for i in range(order):
        # swap current and previous coefficients
        tmp = curr
        curr = prev
        prev = tmp

        # Recurse
        k = ac[i+1]
        for j in range(i):
            k -= prev[j] * ac[i-j]
        curr[i] = k / error
        error *= 1 - curr[i]**2
        for j in range(i):
            curr[j] = prev[j] - curr[i] * prev[i-j-1]
    return curr

# Levinson-Durbin recursion to calculate reflection coefficients from
# autocorrelaton.
def ARLevinson(ac, order=10):
    if ac.ndim > 1:
        ret = np.ndarray((ac.shape[0], order))
        gain = np.ndarray(ac.shape[0])
        for f in range(ac.shape[0]):
            ret[f], gain[f] = ARLevinson(ac[f], order)
        return ret, gain
    
    coef = levinson(ac, order)

    # Actually calculates gain squared
    gain = ac[0] - np.dot(coef, ac[1:order+1])
    return coef, gain

# Ridge regression implementation of AR
def ARRidge(ac, order=10, ridge=0.0):
    if ac.ndim > 1:
        ret = np.ndarray((ac.shape[0], order))
        gain = np.ndarray(ac.shape[0])
        for f in range(ac.shape[0]):
            ret[f], gain[f] = ARRidge(ac[f], order, ridge)
        return ret, gain
    
    coef = levinson(ac, order, ridge*ac[0])
    gain = ac[0] - np.dot(coef, ac[1:order+1])
    #coef = levinson(ac, order, ridge*gain)
    #gain = ac[0] - np.dot(coef, ac[1:order+1])
    return coef, gain

# Lasso-like implementation of AR
def ARLasso(ac, order=10, ridge=0.0):
    if ac.ndim > 1:
        ret = np.ndarray((ac.shape[0], order))
        gain = np.ndarray(ac.shape[0])
        for f in range(ac.shape[0]):
            ret[f], gain[f] = ARLasso(ac[f], order, ridge)
        return ret, gain
    
    # Convert ac into matrices
    YY = np.ndarray((order, order))
    Yy = np.ndarray(order)
    for i in range(order):
        Yy[i] = ac[i+1] * ac.size
        for j in range(order):
            YY[i,j] = ac[abs(i-j)] * ac.size

    # Initialise lasso with ridge
    gain = ac[0]
    A = np.zeros((order, order))
    for i in range(order):
        A[i,i] = ridge*ac[0]*ac.size
    coef = np.dot(linalg.inv(YY+A), Yy)

    for i in range(10):
        for j in range(order):
            A[j,j] = np.sqrt(abs(coef[j]))
        gain = ac[0] + np.dot(coef, (np.dot(YY, coef) - 2*Yy)) / ac.size
        B = np.identity(order) * gain
        X = linalg.inv(np.dot(A, np.dot(YY, A)) + ridge*B)
        coef = np.dot(np.dot(A, np.dot(X, A)), Yy)

    return coef, gain

# AR power spectrum
def ARSpectrum(a, g, nSpec=256, twiddle=None):
    if twiddle is None:
        # Pre-compute the "twiddle" factors; saves a lot of CPU
        twiddle = np.ndarray((nSpec,a.shape[a.ndim-1]), dtype='complex')
        for i in range(nSpec):
            for j in range(twiddle.shape[1]):
                twiddle[i,j] = np.exp(-1j * np.pi * i * (j+1) / nSpec)
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], nSpec))
        for f in range(a.shape[0]):
            ret[f] = ARSpectrum(a[f], g[f], nSpec, twiddle)
        return ret
    
    spec = np.ndarray(nSpec)
    for i in range(nSpec):
        sm = np.dot(a,twiddle[i])
        spec[i] = g / abs(1 - sm)**2
    return spec

# AR cepstrum
def ARCepstrum(a, g, nCep=12):
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], nCep+1))
        for f in range(a.shape[0]):
            ret[f] = ARCepstrum(a[f], g[f], nCep)
        return ret

    cep = np.ndarray(nCep+1)
    for i in range(nCep):
        sum = 0
        for k in range(i):
            index = i-k-1
            if (index < a.size):
                sum += a[index] * cep[k] * (k+1)
        cep[i] = sum / (i+1)
        if (i < a.size):
            cep[i] += a[i]
    cep[nCep] = np.log(max(g, 1e-8))
    return cep


# Alpha values for mel scale at various frequencies
mel = {
    8000: 0.31,
    10000: 0.35,
    12000: 0.37,
    16000: 0.42,
    20000: 0.44,
    22050: 0.45
}

# Alpha values for bark scale at various frequencies
bark = {
    8000: 0.42,
    10000: 0.47,
    12000: 0.50,
    16000: 0.55
}

# Bilinear transform
def BilinearWarpOppenheim(a, alpha=0.0, size=None):
    isize = a.shape[a.ndim-1]
    if size is None:
        osize = isize
    else:
        osize = size

    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], osize))
        for f in range(a.shape[0]):
            ret[f] = BilinearWarpOppenheim(a[f], alpha, size)
        return ret
    
    # Oppenheim's recursion; very slow
    t = np.zeros(osize)
    y = np.zeros(osize)
    alpha2 = 1-alpha**2
    for k in range(isize-1,-1,-1):
        t[0] = alpha * y[0] + a[k]
        t[1] = alpha * y[1] + alpha2 * y[0]
        for n in range(2,osize):
            t[n] = y[n-1] + alpha * (y[n] - t[n-1])

        tmp = y
        y = t
        t = tmp

    return y

# Bilinear warp of AR coefficients that exclude a[0], i.e., we need to
# prepend a 1 for the standard warp to work.
def ARBilinearWarp(a, g, alpha=0, matrix=None):
    if matrix is None:
        m = BilinearWarpMatrix(a.shape[a.ndim-1]+1, alpha)
    else:
        m = matrix
    if a.ndim > 1:
        reta = np.ndarray(a.shape)
        retg = np.ndarray(g.shape)
        for f in range(a.shape[0]):
            reta[f], retg[f] = ARBilinearWarp(a[f], g[f], alpha, m)
        return reta, retg

    # In the AR case, we need to prepend a 1
    a = np.dot(m, np.insert(a, 0, 1))
    g /= a[0]
    a /= a[0]
    return a[1:], g

# Oppenheim's recursion expressed as a matrix.
def BilinearWarpMatrix(n, alpha=0.0, size=None):
    if size is None:
        rows = n
    else:
        rows = size

    a = np.ndarray((rows, n))
    for j in range(n):
        a[0,j] = alpha**j
    for i in range(1, rows):
        a[i,0] = 0
    for i in range(1, rows):
        for j in range(1, n):
            a[i,j] = a[i-1,j-1] + alpha * (a[i,j-1] - a[i-1,j])

    return a

# HTK parameter kinds
parmKind = {
    "LPC":       1,
    "LPCEPSTRA": 3,
    "MFCC":      6,
    "FBANK":     7,
    "MELSPEC":   8,
    "USER":      9,
    "PLP":      11,
    "E":   0000100,
    "N":   0000200,
    "D":   0000400,
    "A":   0001000,
    "Z":   0004000,
    "0":   0020000,
    "T":   0100000
}

# Sink to HTK file
from struct import pack
import array
def HTKSink(fileName, a, period=0.01, kind="USER"):
    if (a.ndim != 2):
        print "Dimension must be 2"
        exit(1)

    htkKind = 0
    for k in kind.split('_'):
        htkKind |= parmKind[k]
    htkPeriod = period * 1e7 + 0.5
    header = pack('>iihh', a.shape[0], htkPeriod, a.shape[1]*4, htkKind)

    # Need to convert ndarray to array to write as 4 byte.  You'd
    # think ndarray.tofile would do that, but it just casts to double.
    with open(fileName, 'wb') as f:
        f.write(header)
        v = np.array(a, dtype='f').byteswap()
        v.tofile(f)
