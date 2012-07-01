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

# Parameters!
# Get an environment variable
from os import environ
def Parameter(param, default=None):
    if param in environ:
        print 'export {0}={1}'.format(param, environ[param])
        # Try to cast it to a numeric type
        for caster in (int, float):
            try:
                return caster(environ[param])
            except ValueError:
                pass
        return environ[param]
    else:
        # Otherwise it's whatever was supplied as a default
        print '# export {0}={1}'.format(param, default)
        return default

def newshape(s, lowdim=0):
    """
    Given a shape s, calculate the shape of a new array with the
    lowest dimension replaced.  If lowdim is zero the lowest dimension
    is removed, otherwise it is replaced.
    """
    sl = list(s)
    if lowdim == 0:
        sl.pop()
    else:
        sl[-1] = lowdim
    return tuple(sl)


def shapeiter(shape, dim=0, index=None):
    """
    Given a tuple representing an ndarray shape, iterates over all
    indices in that shape in C order.  Typically it should be called
    as shapeiter(shape); the other arguments are part of the recursive
    function.
    """
    if not index:
        index = [0]*len(shape)
    for i in range(shape[dim]):
        index[dim] = i
        if dim < len(shape)-1:
            for val in shapeiter(shape, dim+1, index):
                yield val
        else:
            yield tuple(index)

def refiter(a, shape):
    """
    Iterates over a shape using shapeiter(), but uses the indices to
    yield references into the arrays passed in a.
    """
    if len(shape) == 0:
        yield a
    else:
        for i in shapeiter(shape):
            if type(a) == list:
                yield [x[i] for x in a]
            else:
                yield a[i]


# Convert between frequency and things
def hertz_to_dftbin(hz, fs, rate):
    return int(float(hz) / float(rate) * fs + 0.5)

def dftbin_to_hertz(b, fs, rate):
    return float(b) * rate / float(fs)

def seconds_to_acbin(sec, rate):
    return int(float(sec) * float(rate) + 0.5)

def acbin_to_seconds(b, rate):
    return float(b) / rate

#
# The functions here use ThisFormat rather than this_format to make
# them look more like Tracter.  They also use the lowdims() iterator
#

# Load a .wav file
def WavSource(file):
    rate, audio = wavfile.read(file)
    if audio.dtype == 'int16':
        audio = np.cast['float'](audio)
        audio /= 32768
    return rate, audio

# Save a .wav file
def WavSink(a, file, rate):
    audio = a * 32768
    audio = np.cast['int16'](audio)
    wavfile.write(file, rate, audio)
    return a

# Filter comprising a single zero
def ZeroFilter(a, zero=0.97):
    filter = np.zeros(a.size)
    store = a[0]
    for i in range(a.size):
        filter[i] = a[i] - zero * store
        store = a[i]
    return filter

def PoleFilter(a, pole=0.97):
    """
    Single pole filter
    """
    filter = np.zeros(a.size)
    store = 0.0
    for i in range(a.size):
        filter[i] = a[i] + pole * store
        store = filter[i]
    return filter
    

# Convert array to framed array
# The opposite of this for size = period is a.flatten
def Frame(a, size=512, period=256, pad=True):
    if pad:
        # This ensures that frames are aligned in the centre
        x = [a[0]]*(size/2)
        y = [a[-1]]*(size/2)
        a = np.concatenate((x, a, y))
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
        e[r] = linalg.norm(a[r])**2
    return e

# Lx norm
def Norm(a, L=2):
    out = np.ndarray(newshape(a.shape))
    for i, o in lowdims(a, out):
        o[0] = linalg.norm(i, ord=L)
    return out

# Zero mean a framed signal
def ZeroMean(a):
    ret = np.ndarray(a.shape)
    for i, o in refiter([a, ret], newshape(a.shape)):
        o[...] = i - np.mean(i)
    return ret

def Periodogram(a):
    size = a.shape[-1]/2 + 1
    ret = np.ones(newshape(a.shape, size))
    for i, o in refiter([a, ret], newshape(a.shape)):
        o[...] = np.abs(np.fft.fft(i)[:size])**2
    return ret

def Harmonogram(a, input=None, norm=False):
    if input == 'psd':
        size = a.shape[-1]
    else:
        size = a.shape[-1]/2 + 1
    ret = np.zeros(newshape(a.shape, size))

    for i, o in refiter([a, ret], newshape(a.shape)):
        if input == 'psd':
            psd = i
        else:
            psd = np.abs(np.fft.fft(i))**2
        o[0] = psd[0]
        for b in range(1, len(o)):
            h = b
            n = 0
            while h < len(o):
                o[b] += psd[h]
                h *= 2
                n += 1
            if norm:
                o[b] /= n

    return ret


def Autocorrelation(a, input=None):
    """
    Calculate autocorrelation.  Default is assume framed time samples
    as input, but input=psd indicates periodogram input.  Returns an
    array of size len(a)/2+1, or len(a) if a is a psd.
    """
    if input == 'psd':
        size = a.shape[-1]
    else:
        size = a.shape[-1]/2 + 1
    ret = np.ndarray(newshape(a.shape, size))

    for i, o in refiter([a, ret], newshape(a.shape)):
        if input == 'psd':
            dpsd = np.append(i, i[-2:0:-1])
        else:
            dpsd = abs(np.fft.fft(i))**2
        dft = np.fft.ifft(dpsd)[:size]
        o[...] = np.real(dft)/dpsd.size
    return ret

#
# All of the methods below actually calculate gain squared
#
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
        Y = Frame(a[:a.size-1], size=order, period=1, pad=False)
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

# AR excitation filter
def ARExcitation(a, ar, g):
    if a.ndim > 1:
        ret = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret[f] = ARExcitation(a[f], ar[f], g[f])
        return ret

    c = np.append(-ar[::-1], 1)
    r = np.ndarray(len(a))
    for i in range(len(a)):
        if i < len(c):
            r[i] = np.dot(a[:i+1], c[-i-1:]) / np.sqrt(g)
        else:
            r[i] = np.dot(a[i-len(c)+1:i+1], c) / np.sqrt(g)
    return r

# AR resynthesis filter
def ARResynthesis(e, ar, g):
    if e.ndim > 1:
        ret = np.ndarray(e.shape)
        for f in range(e.shape[0]):
            ret[f] = ARResynthesis(e[f], ar[f], g[f])
        return ret

    c = ar[::-1]
    r = np.ndarray(len(e))
    g = np.sqrt(g)
    r[0] = e[0]*g
    for i in range(1,len(e)):
        if i < len(c):
            r[i] = e[i]*g + np.dot(r[:i], c[-i:])
        else:
            r[i] = e[i]*g + np.dot(r[i-len(c):i], c)
    return r

# Sparse AR analysis
def ARSparse(a, order=10):
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], order))
        gain = np.ndarray(a.shape[0])
        for f in range(a.shape[0]):
            ret[f], gain[f] = ARSparse(a[f], order)
        return ret, gain

    # Follow the matrix based method to the letter.  elop contains the
    # poles reversed, coef is the poles in order.
    # Initialise with ML
    gamma = 1
    X = np.identity(len(a)-order)
    for iter in range(5):
        Y = np.dot(X, Frame(a[:a.size-1], size=order, period=1, pad=False))
        y = np.dot(X, a[order:])
        YY = np.dot(Y.T,Y)
        Yy = np.dot(Y.T,y)
        elop = np.dot(linalg.inv(YY), Yy)
        coef = elop[::-1]
        gain = gamma * (np.dot(y,y) - np.dot(elop,Yy)) / y.size
        exn = ARExcitation(a, coef, gain)
        #print iter, linalg.norm(exn[order:], ord=1)
        for i in range(len(exn)):
            exn[i] = max(abs(exn[i]),1e-6)
        X = np.diag(1 / np.sqrt(exn[order:]))
        gamma = np.sqrt(2)
        #gamma = 2

    return (coef, gain)

# Solve polynomial corresponding to AR solution
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

# AllPass transform
def AllPassWarpOppenheim(a, alpha=0.0, size=None):
    isize = a.shape[a.ndim-1]
    if size is None:
        osize = isize
    else:
        osize = size

    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], osize))
        for f in range(a.shape[0]):
            ret[f] = AllPassWarpOppenheim(a[f], alpha, size)
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

# AllPass warp of AR coefficients that exclude a[0], i.e., we need to
# prepend a 1 for the standard warp to work.
def ARAllPassWarp(a, g, alpha=0, matrix=None):
    if matrix is None:
        m = AllPassWarpMatrix(a.shape[a.ndim-1]+1, alpha)
    else:
        m = matrix
    if a.ndim > 1:
        reta = np.ndarray(a.shape)
        retg = np.ndarray(g.shape)
        for f in range(a.shape[0]):
            reta[f], retg[f] = ARAllPassWarp(a[f], g[f], alpha, m)
        return reta, retg

    # In the AR case, we need to prepend a 1
    wa = np.dot(m, np.insert(-a, 0, 1))
    wg = g / wa[0]
    wa /= wa[0]
    return -wa[1:], wg

# AllPass warp of autocorrelation
def AutocorrelationAllPassWarp(a, alpha=0, size=None, matrix=None):

    # Rows defaults to 
    if size is None:
        rows = a.shape[-1]
    else:
        rows = size

    # Precompute a warping matrix
    if matrix is None:
        m = AllPassWarpMatrix(a.shape[-1], alpha, rows)
    else:
        m = matrix

    if a.ndim > 1:
        s = list(a.shape)
        s[-1] = rows
        reta = np.ndarray(s)
        for f in range(a.shape[0]):
            reta[f] = AutocorrelationAllPassWarp(a[f], alpha, rows, m)
        return reta

    # So, here is a single autocorrelation
    a[0] /= 2
    wa = np.dot(m, a)
    wa[0] *= 2
    return wa

# Oppenheim's recursion expressed as a matrix.
def AllPassWarpMatrix(n, alpha=0.0, size=None):
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
    "LPREFC":    2,
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
from os import makedirs
from os.path import dirname, exists
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
    dir = dirname(fileName)
    if dir and not exists(dir):
        makedirs(dir)
    with open(fileName, 'wb') as f:
        f.write(header)
        v = np.array(a, dtype='f').byteswap()
        v.tofile(f)

# Calculate mean; typically cepstral, but it doesn't matter here
def Mean(a):
    # Not quite sure of the meaning for non-2d
    if (a.ndim != 2):
        print "Dimension must be 2"
        exit(1)

    return np.mean(a, axis=0)

# Subtract a vector; typically cepstral mean...
def Subtract(a, m):
    if a.ndim > m.ndim:
        ret = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret[f] = Subtract(a[f], m)
        return ret

    return a-m

# Calculate variance; typically cepstral, but it doesn't matter here
def StdDev(a):
    # Not quite sure of the meaning for non-2d
    if (a.ndim != 2):
        print "Dimension must be 2"
        exit(1)

    return np.std(a, axis=0)

# Divide
def Divide(a, m):
    if a.ndim > m.ndim:
        ret = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret[f] = Divide(a[f], m)
        return ret

    return a/m

# Noise estimate; half at one end, half at the other
def Noise(a, frames=10):
    if a.ndim != 2:
        print "Dimension must be 2"
        exit(1)

    f = frames / 2
    sum1 = np.sum(a[:f,:], axis=0)
    sum2 = np.sum(a[-f:,:], axis=0)
    return (sum1 + sum2) / frames

# SNR spectrum
def SNRSpectrum(a, n):
    if a.ndim > 1:
        ret = np.ndarray(a.shape)
        for f in range(a.shape[0]):
            ret[f] = SNRSpectrum(a[f], n)
        return ret

    x = np.ndarray((2, n.size))
    x[0] = a/n
    x[1] = np.ones(n.size)
    return x.max(axis=0)

# Window
# It's trivial, but helps the program look good
def Window(a, w):
    return a*w

# General raised cosine window
def raisedCosine(n, a):
    w = np.zeros(n)
    for i in range(n):
        m = -1
        for j in range(len(a)):
            m *= -1
            w[i] += m * a[j] * np.cos(2 * np.pi * i * j / (n-1))
    return w

# Some particular raised cosine windows
# http://en.wikipedia.org/wiki/Window_function
def nuttall(n):
    return raisedCosine(n, (0.355768, 0.487396, 0.144232, 0.012604))

def blackmanharris(n):
    return raisedCosine(n, (0.35875, 0.48829, 0.14128, 0.01168))

def blackmannuttall(n):
    return raisedCosine(n, (0.3635819, 0.4891775, 0.1365995, 0.0106411))

def gaussian(n, sigma=0.5):
    w = np.zeros(n)
    for i in range(n):
        w[i] = np.exp(-0.5 * ( (i-(n-1)/2.0) / (sigma*(n-1)/2.0) )**2)
    return w

import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
def zplot(fig, a):
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    arg = np.angle(a)
    mag = np.abs(a)
    ax.plot(arg, mag, 'r+')
    ax.set_rmax(1.0)

def specplot(ax, a, r):
    ax.imshow(np.transpose(np.log10(a)),
              origin='lower', aspect='auto', cmap='bone')
    ax.set_yticks((0,a.shape[-1]-1))
    ax.set_yticklabels(('0', r/2))

class Figure:
    def __init__(self, rows=1, cols=1):
        self.rows = rows
        self.cols = cols
        self.next = 1
        self.fig = plt.figure()

    # Plot order: blue, green, red, ...
    def subplot(self):
        if self.next > self.rows * self.cols:
            raise OverflowError('Out of plots')
        axesSubplot = self.fig.add_subplot(self.rows, self.cols, self.next)
        self.next += 1
        return axesSubplot


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

def Argmax(a, loBin=None, hiBin=None):
    """
    Finds the index on the maximum value of each array.  Further,
    ensures that such maxima are not at the borders of the arrays.
    It's not all that great; might be better to find the first trough
    first.
    """
    if not loBin:
        loBin = 0
    if not hiBin:
        hiBin = a.shape[-1]
    ret = np.zeros(newshape(a.shape, 1), dtype='int')
    for i, o in refiter([a, ret], newshape(a.shape)):
        hi = hiBin
        lo = loBin
        m = np.argmax(i[lo:hi])
        while (m == 0 or m == hi-lo) and lo < hi:
            if m == 0:
                lo += 1
            if m == hi-lo:
                hi -= 1
            m = np.argmax(i[lo:hi])
        o[0] = m + lo
    return np.squeeze(ret)

def ACPitch(a, loPitch=40, hiPitch=500, r=16000):
    """
    Finds the pitch contour.  Input should be framed, but not
    windowed.
    """
    fs = a.shape[-1]
    loPeriod = 1.0 / loPitch
    hiPeriod = 1.0 / hiPitch
    loDFTBin = hertz_to_dftbin(loPitch, fs, r)
    hiDFTBin = hertz_to_dftbin(hiPitch, fs, r)
    loACBin = seconds_to_acbin(hiPeriod, r)
    hiACBin = seconds_to_acbin(loPeriod, r)

    # The AC bin for the period of the lowest frequency needs to be
    # smaller than the size of the AC.
    if hiACBin >= fs / 2:
        print "Frame size {0} too small for pitch {1} Hz".format(fs, loPitch)

    # Basic spectral analysis, windowed, for reference.  Don't do
    # pre-emphasis; it will break low F0 speakers.
    win = gaussian(fs)
    wac = Autocorrelation(win)
    wac /= wac[0]

    a = ZeroMean(a)
    w = Window(a, win)

    # Autocorrelation method, loosely after Boersma
    ac = Autocorrelation(w)
    for i in range(len(ac)):
        ac[i] /= ac[i, 0]
    nac = Divide(ac, wac)

    # Pitch bin is the maximum in each frame
    m = Argmax(nac, loACBin, hiACBin)

    # Convert to pitch and harmonic noise ratio
    pitch = np.ndarray(len(m))
    hnr = np.ndarray(len(m))
    var = np.ndarray(len(m))
    prange = hiPitch - loPitch
    for i in range(len(m)):
        pitch[i] = 1.0 / acbin_to_seconds(m[i], r)
        fnac = np.max([nac[i, m[i]], 1e-6])
        hnr[i] = fnac / (1.0 - fnac)
        var[i] = (1.0 / hnr[i] * prange)**2

    # Kalman smoother
    kPitch, kVar = kalman(pitch, var, 1e3, loPitch + prange/2, prange**2)

    # Now run it again, but with tighter limits.  Note that nac can be
    # less than zero any time, and greater than one depending upon
    # floating accuracy.
    mpitch = np.mean(kPitch)
    for i in range(len(nac)):
        hi = seconds_to_acbin(1.0 / (kPitch[i] * 0.75), r)
        lo = seconds_to_acbin(1.0 / (kPitch[i] * 1.5), r)
        rng = hi - lo
        loBin = np.max([lo, loACBin])
        hiBin = np.min([hi, hiACBin])
        m[i] = np.argmax(nac[i, loBin:hiBin]) + loBin
        pitch[i] = 1.0 / acbin_to_seconds(m[i], r)
        fnac = np.max([nac[i, m[i]], 1e-6])
        fnac = np.min([fnac, 1.0 - 1e-6])
        hnr[i] = fnac / (1.0 - fnac)
        var[i] = (1.0 / hnr[i] * rng)**2
    
    # Kalman smoother again
    kPitch, kVar = kalman(pitch, var, 1e8, mpitch, prange**2)

    return kPitch, hnr

def OverlapAdd(a):
    step = a.shape[-1]/2
    ret = np.zeros(step * (a.shape[0]+1))
    for i in range(a.shape[0]):
        x = step*i
        y = x + 2*step
        ret[x:y] += a[i]
    return ret

class Harmonics():
    def __init__(self, rate, order):
        self.rate = float(rate)
        self.order = order
        self.phase = 0.0
        self.twopi = 2.0 * np.pi

    def sample(self, freq, n):
        phi = np.ndarray((self.order, n))
        ret = np.ndarray((n))
        for i in range(n):
            ph = freq / self.rate * self.twopi
            fphi = self.phase + ph * i
            for l in range(self.order):
                phi[l, i] = fphi * (l+1)
        self.phase = phi[0, -1]
        if self.phase > self.twopi:
            self.phase = self.phase % self.twopi

        phi = np.cos(phi) / self.order * np.sqrt(2)
        for i in range(self.order):
            phi[i] *= 2.0 - 2.0 * i / self.order
        return phi.sum(axis=0)

def pulse(n, ptype='impulse', derivative=True):
    pulse = np.zeros((n))
    T = float(n)
    if ptype == 'impulse':
        pulse[T/2] = 1
    elif ptype == 'poly':
        Tp = int(T * 0.4)
        Tn = int(T * 0.16)
        for i in range(Tp):
            t = float(i)
            pulse[i] = 3.0*(t/Tp)**2 - 2.0*(t/Tp)**3
        for i in range(Tn):
            t = float(i)
            pulse[i+Tp] = 1.0 - (t/Tn)**2
    elif ptype == 'trig':
        Tp = int(T * 0.4)
        Tn = int(T * 0.16)
        for i in range(Tp):
            t = float(i)
            pulse[i] = 0.5*(1.0-np.cos(np.pi*t/Tp))
        for i in range(Tn):
            t = float(i)
            pulse[i+Tp] = np.cos(t/Tn * np.pi/2)
    elif ptype == 'gamma':
        alpha = 2
        beta  = 0.5/(alpha-1)
        pulse = np.ndarray((T))
        for i in range(int(T)):
            t = 1.0-float(i)/T
            pulse[i] = np.exp((alpha-1.0)*np.log(t) - t/beta)
    elif ptype == 'igamma':
        alpha = 10
        beta  = 0.16*(alpha+1)
        pulse = np.ndarray((T))
        for i in range(int(T)):
            t = 1.0-float(i)/T
            pulse[i] = np.exp(-(alpha+1.0)*np.log(t) - beta/t)
    elif ptype == 'lf':
        Tp = 0.6 * T
        Te = Tp + 0.1 * T
        Ta = 0.15 * T
        alpha = 15.0/T
        omega = np.pi / Tp
        for i in range(int(Te)):
            t = float(i)
            pulse[i] = (
                -1.0 / np.sin(omega*Te) * np.exp(alpha*(t-Te)) * np.sin(omega*t)
                 )
        for i in range(int(Te), n):
            t = float(i)
            pulse[i] = (
                -(np.exp(-(t-Te)/Ta) - np.exp(-(T-Te)/Ta))
                )
    else:
        raise LookupError('Unknown pulse type ' + ptype)

    if ptype != 'impulse':
        if derivative and ptype != 'lf':
            pulse = ZeroFilter(pulse)
        pulse /= linalg.norm(pulse)

    return pulse

def pulse_response(ptype='impulse', period=100, order=18):
    # Build a pulse train and find the autocorrelation
    w = 1024
    p = pulse(period, ptype)
    p = np.tile(p, w/period+1)
    p = Window(p[:w], np.hanning(w))
    ac = Autocorrelation(p)
    a, g = ARLevinson(ac, order=order)
    return a, g
