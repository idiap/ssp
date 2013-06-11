#
# Copyright 2012 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, December 2012
#
import numpy as np
import scipy.signal as sp
import numpy.linalg as linalg
import ssp

class Autoregression:
    """
    Class containing autoregression methods; requires an order.  The
    word is taken to be one word 'autoregression', but abbreviated to 'ar'.
    """
    def __init__(self, order):
        self.order = int(order)

# All of the methods below actually calculate gain squared.
def ARMatrix(a, order=10, method='matrix'):
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], order))
        gain = np.ndarray(a.shape[0])
        for f in range(a.shape[0]):
            ret[f], gain[f] = ARMatrix(a[f], order, method)
        return ret, gain

    coef = np.zeros(order)

    # Follow the matrix based method to the letter.  elop contains the
    # poles reversed, coef is the poles in order.
    if method == 'matrix':
        Y = ssp.Frame(a[:a.size-1], size=order, period=1, pad=False)
        y = a[order:]

        YY = np.dot(Y.T,Y)
        Yy = np.dot(Y.T,y)
        elop = np.dot(linalg.inv(YY), Yy)
        for i in range(order):
            coef[i] = elop[order-i-1]
        gain = (np.dot(y,y) - np.dot(elop,Yy)) / y.size

    # Use the autocorrelation to populate the matrices.  Here, Yy runs
    # in ascending index, so we get coef in order right away.
    elif method == 'acmatrix':
        ac = Autocorrelation(a)
        YY = np.ndarray((order, order))
        Yy = np.ndarray(order)
        for i in range(order):
            Yy[i] = ac[i+1] * a.size
            for j in range(order):
                YY[i,j] = ac[abs(i-j)] * a.size
        coef = np.dot(linalg.inv(YY), Yy)
        gain = (ac[0] - np.dot(coef,Yy / a.size))
    else:
        print "Unknown AR method"
        exit(1)

    return (coef, gain)

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
    gain = ac[0] - np.dot(coef, ac[1:order+1])
    return coef, gain

# Convert ac into matrices
def ACToMatrix(ac, order):
    YY = np.ndarray((order, order))
    Yy = np.ndarray(order)
    for i in range(order):
        Yy[i] = ac[i+1] * ac.size
        for j in range(order):
            YY[i,j] = ac[abs(i-j)] * ac.size
    return YY, Yy

# Ridge regression implementation of AR
def ARRidge(ac, order=10, ridge=0.0):
    if ac.ndim > 1:
        ret = np.ndarray((ac.shape[0], order))
        gain = np.ndarray(ac.shape[0])
        for f in range(ac.shape[0]):
            ret[f], gain[f] = ARRidge(ac[f], order, ridge)
        return ret, gain

    coef = levinson(ac, order, ridge*ac[0])
    YY, Yy = ACToMatrix(ac, order)
    gain = ac[0] + np.dot(coef, (np.dot(YY, coef) - 2*Yy)) / ac.size
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
    YY, Yy = ACToMatrix(ac, order)

    # Initialise lasso with ridge
    gain = ac[0]
    A = np.zeros((order, order))
    for i in range(order):
        #A[i,i] = ridge*ac[0]*ac.size
        A[i,i] = 0.01*ac[0]
    coef = np.dot(linalg.inv(YY+A), Yy)

    for i in range(10):
        for j in range(order):
            A[j,j] = np.sqrt(abs(coef[j]))
        gain = ac[0] + np.dot(coef, (np.dot(YY, coef) - 2*Yy)) / ac.size
        B = np.identity(order) * gain
        X = linalg.inv(np.dot(A, np.dot(YY, A)) + ridge*B)
        coef = np.dot(np.dot(A, np.dot(X, A)), Yy)
        # Each iteration should reduce the L1 norm of coef
        #print i, linalg.norm(coef, ord=1)

    return coef, gain

# AR power spectrum
# Old version before I found scipy.signal.freqz()
#def ARSpectrum(a, g, nSpec=256, twiddle=None):
#    if twiddle is None:
#        # Pre-compute the "twiddle" factors; saves a lot of CPU
#        twiddle = np.ndarray((nSpec,a.shape[a.ndim-1]), dtype='complex')
#        for i in range(nSpec):
#            for j in range(twiddle.shape[1]):
#                twiddle[i,j] = np.exp(-1.j * np.pi * i * (j+1) / nSpec)
#    if a.ndim > 1:
#        ret = np.ndarray((a.shape[0], nSpec))
#        for f in range(a.shape[0]):
#            ret[f] = ARSpectrum(a[f], g[f], nSpec, twiddle)
#        return ret
#
#    spec = np.ndarray(nSpec)
#    for i in range(nSpec):
#        sm = np.dot(a,twiddle[i])
#        spec[i] = g / abs(1.0 - sm)**2
#    return spec

def ARSpectrum(ar, gg, nSpec=256):
    """
    Wrapper around scipy.signal.freqz() that both converts ar to
    filter coefficients and broadcasts over an array.
    """
    ret = np.ndarray(ssp.newshape(ar.shape, nSpec))
    for a, g, r in ssp.refiter([ar, gg, ret], ssp.newshape(ar.shape)):
        numer = np.sqrt(g)
        denom = -np.insert(a, 0, -1)
        tmp, r[...] = np.abs(sp.freqz(numer, denom, nSpec))**2
    return ret

# AR cepstrum
def ARCepstrum(a, g, order=None):
    if not order:
        order = a.shape[-1]
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], order+1))
        for f in range(a.shape[0]):
            ret[f] = ARCepstrum(a[f], g[f], order)
        return ret

    cep = np.ndarray(order+1)
    for i in range(order):
        sum = 0
        for k in range(i):
            index = i-k-1
            if (index < a.size):
                sum += a[index] * cep[k] * (k+1)
        cep[i] = sum / (i+1)
        if (i < a.size):
            cep[i] += a[i]
    cep[order] = np.log(max(g, 1e-8))
    return cep

# The opposite recursion: cepstrum to AR coeffs
def ARCepstrumToPoly(cep, order=None):
    """
    Convert cepstra to AR polynomial
    """
    if not order:
        order = cep.shape[-1]-1
    ar = np.ndarray(ssp.newshape(cep.shape, order))
    ag = np.ndarray(ssp.newshape(cep.shape, 1))
    for c, a, g in ssp.refiter([cep, ar, ag], ssp.newshape(cep.shape)):
        for i in range(order):
            sum = 0
            for k in range(i):
                index = i-k-1
                if (index < a.size):
                    sum += a[index] * c[k] * (k+1)
            a[i] = -sum / (i+1)
            if (i < a.size):
                a[i] += c[i]
        g[0] = np.exp(c[order])
    return ar, ag.reshape(ssp.newshape(cep.shape))

# AR excitation filter
def ARExcitation(a, ar, gg):
    if a.ndim > 1:
        ret = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret[f] = ARExcitation(a[f], ar[f], gg[f])
        return ret

    # Reverse the coeffs; negate, and add a 1 for the current sample
    c = np.append(-ar[::-1], 1)
    r = np.ndarray(len(a))
    g = 1.0 / np.sqrt(gg)
    for i in range(len(a)):
        if i < len(c):
            r[i] = np.dot(a[:i+1], c[-i-1:]) * g
        else:
            r[i] = np.dot(a[i-len(c)+1:i+1], c) * g
    return r

# AR resynthesis filter
def ARResynthesis(e, ar, gg):
    if e.ndim > 1:
        ret = np.ndarray(e.shape)
        for f in range(e.shape[0]):
            ret[f] = ARResynthesis(e[f], ar[f], gg[f])
        return ret

    c = ar[::-1]
    r = np.ndarray(len(e))
    g = np.sqrt(gg)
    r[0] = e[0]*g
    for i in range(1,len(e)):
        if i < len(c):
            r[i] = e[i]*g + np.dot(r[:i], c[-i:])
        else:
            r[i] = e[i]*g + np.dot(r[i-len(c):i], c)
    return r

# AR resynthesis filter; assuming later overlap-add
def ARResynthesis2(e, ar, gg):
    assert(e.ndim == 2) # Should iterate down to this
    ret = np.ndarray(e.shape)
    for f in range(len(e)):
        c = ar[f,::-1]
        g = np.sqrt(gg[f])
        ret[f,0] = e[f,0]*g
        for i in range(1,len(e[f])):
            if i < len(c):
                ret[f,i] = e[f,i]*g + np.dot(ret[f,:i], c[-i:])
                if f >= 1:
                    # Complete using outputs of previous OLA frame
                    k = len(c) - i
                    j = len(e[f])/2
                    ret[f,i] += np.dot(ret[f-1,j-k:j], c[:k])
            else:
                ret[f,i] = e[f,i]*g + np.dot(ret[f,i-len(c):i], c)
    return ret

# Sparse AR analysis assuming the excitation is distributed Laplacian.
def ARSparse(a, order=10, gamma=1.414):
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], order))
        gain = np.ndarray(a.shape[0])
        for f in range(a.shape[0]):
            ret[f], gain[f] = ARSparse(a[f], order, gamma)
        return ret, gain

    # Initialise with the ML solution
    ac = ssp.Autocorrelation(a)
    coef, gain = ARLevinson(ac, order)
    x = 1.0 / np.abs(ARExcitation(a, coef, gain)[order:])

    # Follow the matrix based method to the letter.  elop contains the
    # poles reversed, coef is the poles in order.
    for iter in range(5):
        X = np.diag(np.sqrt(x)) # Actually root of inverse of X
        Y = np.dot(X, ssp.Frame(a[:a.size-1], size=order, period=1, pad=False))
        y = np.dot(X, a[order:])
        YY = np.dot(Y.T,Y)
        Yy = np.dot(Y.T,y)
        elop = np.dot(linalg.inv(YY), Yy)
        coef = elop[::-1]
        xx = ARExcitation(a, coef, 1.0)[order:]**2
        gain = gamma * np.dot(xx,x) / y.size
        #        for i in range(len(exn)):
        #            exn[i] = max(abs(exn[i]),1e-6)
        x = 1.0 / np.sqrt(xx / gain)

    return (coef, gain)

# AR analysis assuming the excitation is distributed Student-t.
def ARStudent(a, order=10, df=1.0):
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], order))
        gain = np.ndarray(a.shape[0])
        for f in range(a.shape[0]):
            ret[f], gain[f] = ARStudent(a[f], order, df)
        return ret, gain

    # Initialise with the ML solution
    ac = ssp.Autocorrelation(a)
    coef, gain = ARLevinson(ac, order)
    x = 1.0 / (ARExcitation(a, coef, gain)[order:]**2 + df)

    # Follow the matrix based method to the letter.  elop contains the
    # poles reversed, coef is the poles in order.
    for iter in range(5):
        X = np.diag(np.sqrt(x)) # Actually root of inverse of X
        Y = np.dot(X, ssp.Frame(a[:a.size-1], size=order, period=1, pad=False))
        y = np.dot(X, a[order:])
        YY = np.dot(Y.T,Y)
        Yy = np.dot(Y.T,y)
        elop = np.dot(linalg.inv(YY), Yy)
        coef = elop[::-1]
        xx = ARExcitation(a, coef, 1.0)[order:]**2
        gain = (df+1) * np.dot(xx,x) / y.size
        x = 1.0 / (xx / gain + df)

    return (coef, gain)

# Solve polynomial corresponding to AR solution
def ARRoots(a):
    if a.ndim > 1:
        ret = np.ndarray(a.shape, dtype='complex')
        for f in range(a.shape[0]):
            ret[f] = ARRoots(a[f])
        return ret

    # The refection coeffs are negative, so insert -1 and assume that
    # the whole thing can be multiplied by -1
    r = np.roots(np.insert(a, 0, -1))
    return r

# Tries to find the angle corresponding to the fundamental frequency
def ARAngle(a):
    if a.ndim > 1:
        ret_m = np.ndarray(a.shape)
        ret_s = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret_m[f], ret_s[f] = ARAngle(a[f])
        return ret_m, ret_s

    # First extract the angles of large poles above the real line
    t = np.zeros(len(a))
    j = 0
    for i in range(len(a)):
        if np.abs(a[i]) > 0.85 and np.imag(a[i]) > 0:
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

def ARLogLikelihoodRatio(a, order=10):
    if a.ndim > 1:
        ret = np.ndarray(a.shape[0])
        for f in range(a.shape[0]):
            ret[f] = ARLogLikelihoodRatio(a[f], order)
        return ret

    # Usual Gaussian
    ac = Autocorrelation(a)
    coef, gain = ARLevinson(ac, order)
    exn = ARExcitation(a, coef, gain)
    llGauss = - len(exn)/2 * np.log(2*np.pi) - 0.5 * np.dot(exn, exn)

    # Unusual Laplacian
    gamma = 1
    X = np.identity(len(a)-order)
    for iter in range(5):
        Y = np.dot(X, Frame(a[:a.size-1], size=order, period=1))
        y = np.dot(X, a[order:])
        YY = np.dot(Y.T,Y)
        Yy = np.dot(Y.T,y)
        elop = np.dot(linalg.inv(YY), Yy)
        coef = elop[::-1]
        gain = gamma * (np.dot(y,y) - np.dot(elop,Yy)) / y.size
        exn = ARExcitation(a, coef, gain)
        for i in range(len(exn)):
            exn[i] = max(abs(exn[i]),1e-6)
        X = np.diag(1 / np.sqrt(exn[order:]))
        gamma = 2

    exn = ARExcitation(a, coef, gain)
    llLaplace = -gamma * np.sum(np.abs(exn))

    return llLaplace - np.logaddexp(llLaplace, llGauss)

# import numpy.polynomial as pn

def ARHarmonicPoly(f0, rate, mag=0.99):
    """
    Generates coefficients of an AR polynomial corresponding to a
    harmonic excitation at frequency f0
    """
    omega = float(f0)/rate * 2.0 * np.pi
    n = int(np.pi / omega)
    roots = np.ndarray((n*2), dtype='complex')
    for i in range(n):
        a = omega * (i+1)
        c = mag*np.cos(a)
        s = mag*np.sin(a)
        roots[i*2] = complex(c, s)
        roots[i*2+1] = complex(c, -s)

    if True:
        poly = np.poly(roots)[1:]
    else:
        # polyfromroots() returns the trailing coeff = 1, so just
        # knock off that one and reverse the rest as we have negative
        # powers.
        poly = pn.polyfromroots(roots).real[-2::-1]
    if np.max(np.abs(poly)) >= 1.0:
        print roots, np.prod(roots), poly
        raise OverflowError('Poly too big')
    return -poly

def pulse_response(gm, pcm, period=100, order=18):
    # Build a pulse train and find the autocorrelation
    w = 1024
    p = gm.pulse(period, pcm)
    p = np.tile(p, w/period+1)
    p = ssp.Window(p[:w], np.hanning(w))
    ac = ssp.Autocorrelation(p)
    a, g = ARLevinson(ac, order=order)
    return a, g
