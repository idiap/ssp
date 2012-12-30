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
def parameter(param, default=None):
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

# Iterators.  These allow slicing and broadcasting when the default
# rules are not appropriate.
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
    yield references into the arrays passed in a.  Beware: Iterating over a
    one dimensional array yields numbers, not arrays.
    """
    if len(shape) == 0:
        yield a
    else:
        for i in shapeiter(shape):
            if type(a) == list:
                # A list of arrays was passed
                yield [x[i] for x in a]
            else:
                # Just one
                yield a[i]

class PulseCodeModulation:
    """
    A PulseCodeModulation (the instance might be abbreviated to pcm)
    is a class that represents something with a sample rate.  It
    contains methods that require a sample rate in order to work.
    """
    def __init__(self, rate=None):
        self.rate = rate

    def WavSource(self, file):
        """ Reads a wav file into a numpy array """
        rate, audio = wavfile.read(file)
        if audio.dtype == 'int16':
            audio = np.cast['float'](audio)
            audio /= 32768
        if self.rate is None:
            self.rate = rate
        elif self.rate != rate:
            raise ValueError("WavSource: Wrong sample rate")
        return audio

    def WavSink(self, a, file):
        """ Writes a numpy array to a 16 bit wav file """
        audio = a * 32768
        audio = np.cast['int16'](audio)
        wavfile.write(file, self.rate, audio)
        return a

    def speech_ar_order(self):
        """ The rationale here is purely a rule of thumb. """
        if self.rate is None:
            raise ValueError("rate undefined")
        return int(self.rate/1000)+2

    def hertz_to_dftbin(self, hz, fs):
        """ Returns the DFT bin corresponding to a value in Hertz """
        return int(float(hz) / float(self.rate) * fs + 0.5)

    def dftbin_to_hertz(self, b, fs):
        """ Returns the value in Hertz corresponding to a DFT bin """
        return float(b) * self.rate / float(fs)

    def seconds_to_acbin(self, sec):
        """ Returns the autocorrelation bin corresponding to a given lag """
        return int(float(sec) * float(self.rate) + 0.5)

    def acbin_to_seconds(self, b):
        """ Returns the lag corresponding to a given autocorrelation bin """
        return float(b) / self.rate

    def period_to_seconds(self, n):
        """ Returns the time corrensponding to a given number of samples """
        return n / float(self.rate)

    def seconds_to_period(self, sec, power=None):
        """
        Returns a number of samples corresponding to the given time.
        If power is 'atleast' then it is rounded up to the next power
        of 2; 'atmost' rounds down to the previous power of 2.
        """
        period = self.rate * sec
        if power is None:
            return int(period)
            
        # May be faster than log2()
        s = 1
        while s < period:
            s *= 2
        if s == period or power == 'atleast':
            return s
        else:
            return s/2

    def hertz_to_radians(self, hz):
        """ Returns the angle corresponding to a value in Hertz """
        return float(hz) / float(self.rate) * np.pi * 2


class Autoregression:
    """
    Class containing autoregression methods; requires an order.  The
    word is taken to be one word 'autoregression', but abbreviated to 'ar'.
    """
    def __init__(self, order):
        self.order = int(order)


#
# The functions here use ThisFormat rather than this_format to make
# them look more like Tracter.  They also use the lowdims() iterator
#

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

def PolePairFilter(a, mag, angle):
    """
    Takes a magnitude and phase taken to be the location of a pole;
    filters the signal using the implied conjugate pair.
    """
    r1 = 2.0 * mag * np.cos(angle)
    r2 = -mag**2
    filter = np.zeros(a.size)
    for i in range(a.size):
        filter[i] = a[i]
        if i > 0:
            filter[i] += r1 * filter[i-1]
        if i > 2:
            filter[i] += r2 * filter[i-2]
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


def Autocorrelation(a, power=2, input=None):
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
            dpsd = abs(np.fft.fft(i))**power
        dft = np.fft.ifft(dpsd)[:size]
        o[...] = np.real(dft)/dpsd.size
    return ret

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
                twiddle[i,j] = np.exp(-1.j * np.pi * i * (j+1) / nSpec)
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], nSpec))
        for f in range(a.shape[0]):
            ret[f] = ARSpectrum(a[f], g[f], nSpec, twiddle)
        return ret
    
    spec = np.ndarray(nSpec)
    for i in range(nSpec):
        sm = np.dot(a,twiddle[i])
        spec[i] = g / abs(1.0 - sm)**2
    return spec

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
    ar = np.ndarray(newshape(cep.shape, order))
    ag = np.ndarray(newshape(cep.shape, 1))
    for c, a, g in refiter([cep, ar, ag], newshape(cep.shape)):
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
    return ar, ag.reshape(newshape(cep.shape))

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

# Sparse AR analysis
def ARSparse(a, order=10, gamma=1.414):
    if a.ndim > 1:
        ret = np.ndarray((a.shape[0], order))
        gain = np.ndarray(a.shape[0])
        for f in range(a.shape[0]):
            ret[f], gain[f] = ARSparse(a[f], order)
        return ret, gain

    # Follow the matrix based method to the letter.  elop contains the
    # poles reversed, coef is the poles in order.
    
    iter_gamma = 1  # Initialise with ML
    X = np.identity(len(a)-order)
    for iter in range(5):
        Y = np.dot(X, Frame(a[:a.size-1], size=order, period=1, pad=False))
        y = np.dot(X, a[order:])
        YY = np.dot(Y.T,Y)
        Yy = np.dot(Y.T,y)
        elop = np.dot(linalg.inv(YY), Yy)
        coef = elop[::-1]
        gain = iter_gamma * (np.dot(y,y) - np.dot(elop,Yy)) / y.size
        exn = ARExcitation(a, coef, gain)
        #print iter, linalg.norm(exn[order:], ord=1)
        for i in range(len(exn)):
            exn[i] = max(abs(exn[i]),1e-6)
        X = np.diag(1 / np.sqrt(exn[order:]))
        iter_gamma = gamma  # Continue iteration with supplied gamma

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
from struct import pack, unpack
from os import makedirs
from os.path import dirname, exists
def HTKSink(fileName, a, period=0.01, kind="USER"):
    if (a.ndim != 2):
        print "Dimension must be 2"
        exit(1)

    htkKind = 0
    for k in kind.split('_'):
        htkKind |= parmKind[k]
    htkPeriod = period * 1e7 + 0.5
    header = pack('>iihh', a.shape[0], htkPeriod, a.shape[1]*4, htkKind)

    dir = dirname(fileName)
    if dir and not exists(dir):
        makedirs(dir)
    with open(fileName, 'wb') as f:
        # Need to create a new array here of type float32 ('f') such
        # that it is written as 4 bytes.  Casting doesn't seem to
        # work.
        f.write(header)
        v = np.array(a, dtype='f').byteswap()
        v.tofile(f)

# Source from HTK file
def HTKSource(fileName):
    with open(fileName, 'r') as f:
        # The resulting array here is float32.  We could explicitly
        # cast it to double, but that will happen further up in the
        # program anyway.
        header = f.read(12)
        (htkSize, htkPeriod, htkVecSize, htkKind) = unpack('>iihh', header)
        data = np.fromfile(f, dtype='f')
        a = data.reshape((htkSize, htkVecSize / 4)).byteswap()
    return a, htkPeriod * 1e-7

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
        while (m == 0 or m == hi-lo) and lo+1 < hi:
            if m == 0:
                lo += 1
            if m == hi-lo:
                hi -= 1
            m = np.argmax(i[lo:hi])
        o[0] = m + lo
    return np.squeeze(ret)

def ACPitch(a, pcm, loPitch=40, hiPitch=500):
    """
    Finds the pitch contour.  Input should be framed, but not
    windowed.
    """
    fs = a.shape[-1]
    loPeriod = 1.0 / loPitch
    hiPeriod = 1.0 / hiPitch
    loDFTBin = pcm.hertz_to_dftbin(loPitch, fs)
    hiDFTBin = pcm.hertz_to_dftbin(hiPitch, fs)
    loACBin = pcm.seconds_to_acbin(hiPeriod)
    hiACBin = pcm.seconds_to_acbin(loPeriod)

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
        pitch[i] = 1.0 / pcm.acbin_to_seconds(m[i])
        fnac = np.max([nac[i, m[i]], 1e-6])
        if (nac[i, m[i]-1] > nac[i, m[i]]) or (nac[i, m[i]+1] > nac[i, m[i]]):
            # No peak found; set HNR small
            hnr[i] = 1e-8
        else:
            hnr[i] = fnac / (1.0 - fnac)
        var[i] = (1.0 / hnr[i] * prange)**2

    # Kalman smoother
    kPitch, kVar = kalman(pitch, var, 1e3, loPitch + prange/2, prange**2)

    # Now run it again, but with tighter limits.  Note that nac can be
    # less than zero any time, and greater than one depending upon
    # floating accuracy.
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
        fnac = np.min([fnac, 1.0 - 1e-6])
        if (nac[i, m[i]-1] > nac[i, m[i]]) or (nac[i, m[i]+1] > nac[i, m[i]]):
            # No peak found; set HNR small
            hnr[i] = 1e-8
        else:
            hnr[i] = fnac / (1.0 - fnac)
        var[i] = (1.0 / hnr[i] * rng)**2
    
    # Kalman smoother again
    # 1e4 is about right for < 10ms frame shift.  It was 1e8 for ~25ms
    kPitch, kVar = kalman(pitch, var, 1e4, mpitch, prange**2)

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

class GlottalModel:
    """
    Class implementing various glottal modelling strategies.
    """
    def __init__(self, ptype='impulse', params=None):
        self.ptype = ptype
        if self.ptype == 'lf':
            if params is not None:
                (self.Rg, self.Rk, self.Fa) = params
            else:
                # Defaults tested on emime/EF2
                self.Rg = parameter('Rg', 1.0)
                self.Rk = parameter('Rk', 0.2)
                self.Fa = parameter('Fa', 200)
        elif self.ptype == 'zerofilter':
            if params is not None:
                (self.zero) = params
            else:
                self.zero = parameter('Zero', 0.97)
        elif self.ptype == 'polefilter':
            if params is not None:
                (self.pole) = params
            else:
                self.pole = parameter('Pole', 0.97)
        elif self.ptype == 'polezerofilter':
            if params is not None:
                (self.pole, self.zero) = params
            else:
                self.pole = parameter('Pole', 0.97)
                self.zero = parameter('Zero', 1.0)
        elif self.ptype == 'polepairzerofilter':
            if params is not None:
                (self.pole, self.angle, self.zero) = params
            else:
                self.pole = parameter('Pole', 0.97)
                self.angle = parameter('Angle', 0.0)
                self.zero = parameter('Zero', 1.0)

    def pulse(self, n, pcm, derivative=True):
        n = int(n)
        pulse = np.zeros((n))
        T0 = pcm.period_to_seconds(n) # Fundamental period in seconds
        T = float(n)
        if self.ptype == 'impulse':
            pulse[T/2] = 1
        elif self.ptype == 'mipulse':
            pulse[T/2] = -1
        elif self.ptype == 'dimpulse':
            pulse[T/2] = 1
            pulse[T/2+1] = -1
        elif self.ptype == 'poly':
            Tp = int(T * 0.4)
            Tn = int(T * 0.16)
            for i in range(Tp):
                t = float(i)
                pulse[i] = 3.0*(t/Tp)**2 - 2.0*(t/Tp)**3
            for i in range(Tn):
                t = float(i)
                pulse[i+Tp] = 1.0 - (t/Tn)**2
            if derivative:
                pulse = ZeroFilter(pulse)
        elif self.ptype == 'trig':
            Tp = int(T * 0.4)
            Tn = int(T * 0.16)
            for i in range(Tp):
                t = float(i)
                pulse[i] = 0.5*(1.0-np.cos(np.pi*t/Tp))
            for i in range(Tn):
                t = float(i)
                pulse[i+Tp] = np.cos(t/Tn * np.pi/2)
            if derivative:
                pulse = ZeroFilter(pulse)
        elif self.ptype == 'gamma':
            Rg = 1.2
            Rk = 0.3
            Tp = T/(2.0*Rg)
            Te = Tp*(Rk+1.0)
            alpha = 3.0
            beta  = 0.16/(alpha-1)
            for i in range(int(Te)):
                t = 1.0-float(i)/Te
                pulse[i] = np.exp((alpha-1.0)*np.log(t) - t/beta)
            if derivative:
                pulse = ZeroFilter(pulse)
        elif self.ptype == 'igamma':
            alpha = 10
            beta  = 0.16*(alpha+1)
            for i in range(int(T)):
                t = 1.0-float(i)/T
                pulse[i] = np.exp(-(alpha+1.0)*np.log(t) - beta/t)
            if derivative:
                pulse = ZeroFilter(pulse)
        elif self.ptype == 'lf':
            # These three are all in seconds
            Ta = 1.0/(2.0*np.pi*self.Fa)
            Tp = T0/(2.0*self.Rg)
            Te = Tp*(self.Rk+1.0)
            eps = self.lf_epsilon(Te, Ta, T0)
            alpha = self.lf_alpha(Tp, Te, eps, T0)
            omega = np.pi / Tp
            for i in range(int(Te/T0*n+0.5)):
                t = float(i)/n*T0
                pulse[i] = -(
                    np.exp(alpha*t)  * np.sin(omega*t) /
                    np.exp(alpha*Te) / np.sin(omega*Te)
                    )
            for i in range(int(Te/T0*n+0.5), n):
                t = float(i)/n*T0
                pulse[i] = -(
                    (np.exp(-(t-Te)*eps) - np.exp(-(T0-Te)*eps)) /
                    (1.0 - np.exp(-(T0-Te)*eps))
                    )
        elif self.ptype == 'invexp':
            for i in range(n):
                t = 1.0-float(i/T)
                pulse[i] = -np.exp(-t*5)
        elif self.ptype == 'polefilter':
            pulse[0] = 1.0
            pulse = PoleFilter(pulse, self.pole)
        elif self.ptype == 'zerofilter':
            pulse[0] = 1.0
            pulse = ZeroFilter(pulse, self.zero)
        elif self.ptype == 'polezerofilter':
            pulse[1] = 1.0
            pulse = PoleFilter(pulse, self.pole)
            pulse = PoleFilter(pulse, self.pole)
            pulse = ZeroFilter(pulse, self.zero)
        elif self.ptype == 'polepairzerofilter':
            pulse[1] = 1.0
            #pulse = PoleFilter(pulse, params[0])
            pulse = PolePairFilter(pulse, self.pole, self.angle)
            pulse = ZeroFilter(pulse, self.zero)
        elif self.ptype == 'multipulse':
            for i in range(24):
                pulse[i] = 1
        else:
            raise LookupError('Unknown pulse type ' + str(self.ptype))

        # We want the total power in the pulse to average one for each
        # sample, so the power is the length of the pulse
        pulse *= np.sqrt(T) / linalg.norm(pulse)
        return pulse

    def lf_alpha(self, tp, te, epsilon, T0, alpha=0.0):
        """
        Given the three timing parameters and a starting point, uses
        Newton-Raphson to find a value of alpha for the Liljencrants-Fant
        glottal pulse shape.
        """
        tc = T0
        omega = np.pi / tp
        for iter in range(5):
            exp = np.exp(-epsilon*(tc-te))
            esin = np.exp(alpha*te)*np.sin(omega*te)
            f = (
                alpha
                - omega/np.tan(omega*te)
                + omega/esin
                -(alpha**2 + omega**2)*((tc-te)*exp/(1.0-exp) - 1.0/epsilon)
                )
            fd = (
                1.0
                - omega*te/esin
                -2.0*alpha*((tc-te)*exp/(1.0-exp) - 1.0/epsilon)
                )
            alpha -= f/fd
        return alpha

    def lf_epsilon(self, te, ta, T0):
        tce = T0 - te
        epsilon = 1.0/ta
        for iter in range(5):
            f = 1.0 - np.exp(-epsilon * tce) - ta * epsilon
            fd = tce * np.exp(-epsilon * tce) - ta
            epsilon -= f/fd
        
        return epsilon

def pulse_response(gm, pcm, period=100, order=18):
    # Build a pulse train and find the autocorrelation
    w = 1024
    p = gm.pulse(period, pcm)
    p = np.tile(p, w/period+1)
    p = Window(p[:w], np.hanning(w))
    ac = Autocorrelation(p)
    a, g = ARLevinson(ac, order=order)
    return a, g

