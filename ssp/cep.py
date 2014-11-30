#!/usr/bin/python2
#
# Copyright 2013 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, May 2013
#
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

from . import core
from . import ar as AR

def c0(a):
    """
    Compute c0 only.  Does the DFT, but then just sums to get c0
    instead of doing an inverse DFT.
    """
    ldft = np.log(np.abs(np.fft.rfft(a)))
    sum1 = np.sum(ldft)
    sum2 = np.sum(ldft[1:len(ldft)-1])
    return (sum1 + sum2) / len(a)

def zzt(a):
    """
    Brute force route to complex cepstrum via zeros of the
    z-transform.  Could undoubtedly be optimised to construct the
    cepstrum using only half of the (conjugate) poles; but then again
    the big bottleneck is the "roots" call.
    """
    ret = np.ndarray(a.shape)
    for i, o in core.refiter([a, ret], core.newshape(a.shape)):
        r = np.roots(i)
        m = np.abs(r)
        az = []
        bz = []
        for j in range(len(r)):
            if m[j] < 1:
                az.append(r[j])
            else:
                bz.append(1.0/r[j])
        az = np.array(az)
        bz = np.array(bz)
        aaz = az.copy()
        bbz = bz.copy()
        cep = np.zeros((len(i)), dtype='complex')
        for n in range(1, len(i)/2):
            for k in aaz:
                cep[n] -= k
            for k in bbz:
                cep[len(cep)-1-n] -= k
            aaz *= az*(float(n)/(n+1))
            bbz *= bz*(float(n)/(n+1))
        o[...] = np.real(cep)
        o[0] = c0(i)
    return ret

def phase_unwrap(arg):
    """
    Phase unwrapping inspired by Oppenheim & Schafer p 790.  Uses
    epsilon (jump tolerance) of pi,
    cf. http://www.mathworks.ch/ch/help/matlab/ref/unwrap.html The
    linear phase offset is explicitly removed.  It could be done
    instead by shifting the time sequence forward or backwards, but we
    assume there is no need to reconstruct the time sequence.
    """
    eps = np.pi
    threshold = np.pi*2 - eps
    phase = np.ndarray((len(arg)))
    wrap = 0
    phase[0] = arg[0]
    for j in range(1, len(arg)):
        if arg[j] - arg[j-1] > threshold:
            wrap -= 1
        elif arg[j] - arg[j-1] < -threshold:
            wrap += 1
        phase[j] = arg[j] + np.pi * 2 * wrap

    # Remove the linear phase offset
    n = len(phase)
    phase -= np.arange(n) * (phase[-1]-phase[0]) / (n-1) + phase[0]
    return phase

def ComplexCepstrum(a, size=None):
    """
    The complex cepstrum is actually real for a real input sequence.
    If size is given, it is taken to be the size of the resulting
    cepstrum, implemented using a truncation in frequency
    """
    if not size:
        size = a.shape[-1]
    ret = np.zeros(core.newshape(a.shape, size))
    for i, o in core.refiter([a, ret], core.newshape(a.shape)):
        # The fftshift is to center the phase on the center of the
        # time sequence.  This removes an oscillation of +-pi in the
        # dft phase that happens otherwise.
        dft = np.fft.rfft(np.fft.fftshift(i))[:size/2+1]
        arg = phase_unwrap(np.angle(dft))
        logfr = np.log(np.abs(dft), dtype='complex')
        logfr.imag = arg
        o[...] = np.fft.irfft(logfr)
    return ret

def ComplexSpectrum(a, phase=None):
    """
    Given a real input representing a complex cepstrum, return the
    corresponding spectrum.  If phase is 'min' or 'max', return the
    minimum or maximum phase response respectively.
    """
    if phase == 'min':
        # Zero out the negative cepstrum
        tmp = a.copy()
        tmp[:,a.shape[-1]/2:] = 0.0
        tmp[:,0] /= 2
        a = tmp
    elif phase == 'max':
        # Zero out the positive cepstrum
        tmp = a.copy()
        tmp[:,1:a.shape[-1]/2] = 0.0
        tmp[:,0] /= 2
        a = tmp
    ret = np.zeros(core.newshape(a.shape, a.shape[-1]/2+1), dtype='complex')
    for i, o in core.refiter([a, ret], core.newshape(a.shape)):
        dft = np.fft.rfft(i)
        ex = np.exp(dft)
        o[...] = ex
    return ret

def root_complex(r):
    for root in r:
        # Does it contain a complex root?
        if np.abs(root.imag) > 1e-8:
            return True
    return False

def root_negative(r):
    for root in r:
        # Does it contain a negative root?
        if np.real(root) < 0:
            return True
    return False

def root_angle(r):
    """
    Single dimensional version of MinPolar() (below)
    """
    angle = np.pi
    for root in r:
        if np.abs(root.imag) > 1e-8:
            a = np.abs(np.angle(root))
            if a < angle:
                angle = a
    if angle == np.pi:
        angle = 0
    return angle

def MinPolar(c):
    """
    Given an array where shape[-1] is an axis of several complex
    numbers, returns two arrays (magnitude and phase) one dimension
    smaller corresponding to the smallest angle in the array
    """
    sh = core.newshape(c.shape)
    sh1 = core.newshape(c.shape, 1)
    arg = np.ndarray(sh1)
    mag = np.ndarray(sh1)
    for i, a, m in core.refiter([c, arg, mag], sh):
        aMin = np.pi
        mMin = 0.0
        for root in i:
            if np.abs(root.imag) > 1e-8:
                th = np.abs(np.angle(root))
                if th < aMin:
                    aMin = th
                    mMin = np.abs(root)
        if aMin == np.pi:
            aMin = 0
        a[0] = aMin
        m[0] = mMin
    return arg.reshape(sh), mag.reshape(sh)

def glottal_pole(f, pcm, pitch, hnr, visual=False):
    """
    Algorithm to extract glottal poles using complex cepstrum.
    """
    fig = None
    if visual:
        fig = core.Figure(6)
    frame = core.parameter('Frame', 0)

    # Window the framed signal
    frameSize = f.shape[-1]
    w = core.Window(f, core.nuttall(frameSize))
    if visual:
        p = core.Periodogram(w)
        ax1 = fig.subplot()
        fig.specplot(ax1, p, pcm)

    # Excitation - use the windowed frames
    ac = core.Autocorrelation(w)
    ar, gg = AR.ARLevinson(ac, pcm.speech_ar_order())
    ex = AR.ARExcitation(w, ar, gg)

    # Glottal closure.  This should be near the middle as it's
    # windowed.
    # wsize is enough to capture two pitch periods.
    # cwsize is a bigger window, zero-padded version of
    # wsize to allow phase unwrapping.
    mean = np.mean(pitch)
    wsize = pcm.seconds_to_period(1/mean) * 2
    cwsize = 512
    gci = ex.argmax(axis=-1)
    g = np.zeros((len(f), cwsize))
    gw = core.nuttall(wsize)
    for i in range(len(f)):
        beg = gci[i] - wsize/2
        end = gci[i] + wsize/2
        if (beg < 0):
            end += -beg
            beg = 0
        elif (end > frameSize):
            beg -= end-frameSize
            end = frameSize
        # Make sure to window the unwindowed frame
        g[i][cwsize/2-wsize/2:cwsize/2+wsize/2] = gw * f[i, beg:end]

    # Sample frame
    if fig:
        sample = w[frame]
        fr = fig.subplot()
        fr.plot(sample / np.max(np.abs(sample)))
        fr.plot(ex[frame] / np.max(np.abs(ex[frame])))
        fr.set_xlim(0, frameSize)

    # Define a new PCM representing the downsampled signal for complex
    # cepstrum
    cepFreq = core.parameter('CepFreq', 1000.0)
    cepPCM = core.PulseCodeModulation(cepFreq * 2)
    clbin = pcm.hertz_to_dftbin(cepFreq, cwsize)
    if not int(clbin) & 1:
        clbin += 1
    clsize = (clbin-1)*2
    cl = ComplexCepstrum(g, clsize)

    # Maximum phase spectra
    negs = ComplexSpectrum(cl, 'max')

    # Convert negative spectrum to LP
    order = core.parameter('Order', 2)
    negp = np.abs(negs)**2
    ac = core.Autocorrelation(negp, input='psd')
    a, g = AR.ARLevinson(ac, order=order)
    ars = AR.ARSpectrum(a, g, clsize/2+1)
    roots = AR.ARRoots(a)

    if fig:
        neg = fig.subplot()
        fig.specplot(neg, np.abs(negs), cepPCM)
        spec = fig.subplot()
        spec.plot(10*np.log10(np.abs(negs[frame])**2/clsize))
        spec.plot(10*np.log10(np.abs(ars[frame])))

        zpos = roots[frame,0].real
        numer = np.array([1.0, -zpos]) * np.sqrt(g[frame])
        denom = np.insert(-a[frame], 0, 1)
        tmp, zspec = sp.freqz(numer, denom, clsize/2+1)
        spec.plot(10*np.log10(np.abs(zspec)**2))


    # Default pitch range is 40-500 Hz.
    #theta = [root_angle(r) for r in roots]
    theta, magtd = MinPolar(roots)
    loTheta = 1e-8
    hiTheta = cepPCM.hertz_to_radians(500)
    thRange = hiTheta - loTheta
    thVar = (1.0 / hnr * thRange)**2
    for i in range(len(thVar)):
        if theta[i] < loTheta or theta[i] > hiTheta:
            thVar[i] = 1e8
    thMean = cepPCM.hertz_to_radians(mean)
    #                         obs,   obsVar, seqVar,       initMean, initVar
    kTheta, kTVar = core.kalman(theta, thVar, (thMean/4)**2, thMean, thRange**2)

    # Smooth the magnitude
    mVar = (1.0 / hnr)**2
    #                        obs,   obsVar, seqVar, initMean, initVar
    kMag, kMVar = core.kalman(magtd, mVar, 0.01, 0.5, 1.0)

    if fig:
        ang = fig.subplot()
        ang.plot(pitch)
        iCom = []
        vCom = []
        iPos = []
        vPos = []
        for i in range(len(roots)):
            if root_complex(roots[i]):
                iCom.append(i)
                vCom.append(cepPCM.radians_to_hertz(root_angle(roots[i])))
            elif not root_negative(roots[i]):
                iPos.append(i)
                vPos.append(0)
        ang.plot(iCom, vCom, '1')
        ang.plot(iPos, vPos, '2')

        kt = [cepPCM.radians_to_hertz(x) for x in kTheta]
        kv = [cepPCM.radians_to_hertz(x) for x in kTheta+np.sqrt(kTVar)]
        ang.plot(kt)
        ang.plot(kv)

        ang.set_xlim(0,len(roots))
        ang.set_ylim(0, core.parameter('MaxHz', cepFreq))

        magp = fig.subplot()
        magp.plot(magtd)
        magp.plot(kMag)
        magp.plot(kMag + np.sqrt(kMVar))
        magp.plot(kMag - np.sqrt(kMVar))
        #magp.plot(-np.log(magtd+1e-8))
        magp.set_xlim(0,len(magtd))

    if fig:
        pfig = plt.figure()
        pax = pfig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        pos = roots.flatten()
        mag = 1/(np.abs(pos)+1e-8)
        arg = np.angle(pos)
        pax.plot(arg, mag, '3')
        pax.set_rmax(5)

    if fig:
        plt.show()

    return kTheta, kMag, cepPCM

def glottal_pole_lf(f, pcm, pitch, hnr, visual=False):
    """
    Compute glottal pole; return as parameters for the LF model.  The returned
    parameters are defined in the continuous time domain, so they are pcm
    independent.
    """
    kTheta, kMag, cepPCM = glottal_pole(f, pcm, pitch, hnr, visual)
    return kTheta * cepPCM.rate, -np.log(kMag+1e-8) * cepPCM.rate

def glottal_pole_gm(f, pcm, pitch, hnr, visual=False):
    """
    Compute glottal pole; return as parameters for glottal model.  The returned
    parameters are angle and magnitude of a pole, so they are specific to the
    caller's pcm.
    """
    kTheta, kMag, cepPCM = glottal_pole(f, pcm, pitch, hnr, visual)
    scale = cepPCM.rate / pcm.rate
    return kTheta * scale, kMag
