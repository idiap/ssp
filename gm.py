#
# Copyright 2012 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, December 2012
#
import numpy as np
import numpy.linalg as linalg

from ssp import *

def pulse_poly(pulse, Tp, Tn):
    for i in range(Tp):
        t = float(i)
        pulse[i] = 3.0*(t/Tp)**2 - 2.0*(t/Tp)**3
    for i in range(Tn):
        t = float(i)
        pulse[i+Tp] = 1.0 - (t/Tn)**2
    return pulse

def pulse_trig(pulse, Tp, Tn):
    for i in range(Tp):
        t = float(i)
        pulse[i] = 0.5*(1.0-np.cos(np.pi*t/Tp))
    for i in range(Tn):
        t = float(i)
        pulse[i+Tp] = np.cos(t/Tn * np.pi/2)
    return pulse

def pulse_gamma(pulse, Te, alpha, beta):
    for i in range(int(Te)):
        t = 1.0-float(i)/Te
        pulse[i] = np.exp((alpha-1.0)*np.log(t) - t/beta)
    return pulse

def pulse_igamma(pulse, T, alpha, beta):
    for i in range(int(T)):
        t = 1.0-float(i)/T
        pulse[i] = np.exp(-(alpha+1.0)*np.log(t) - beta/t)
    return pulse

def pulse_lf(pulse, T0, Te, alpha, omega, eps):
    """
    Assumes E_e = -1.
    """
    n = len(pulse)
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
    return pulse

def lf_alpha(tp, te, epsilon, T0, alpha=0.0):
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
            - (alpha**2 + omega**2)*((tc-te)*exp/(1.0-exp) - 1.0/epsilon)
            )
        fd = (
            1.0
            - omega*te/esin
            - 2.0*alpha*((tc-te)*exp/(1.0-exp) - 1.0/epsilon)
            )
        alpha -= f/fd
    return alpha

def lf_epsilon(te, ta, T0):
    tce = T0 - te
    epsilon = 1.0/ta
    for iter in range(5):
        f = 1.0 - np.exp(-epsilon * tce) - ta * epsilon
        fd = tce * np.exp(-epsilon * tce) - ta
        epsilon -= f/fd

    return epsilon


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
            pulse_poly(pulse, Tp, Tn)
            if derivative:
                pulse = ZeroFilter(pulse)
        elif self.ptype == 'trig':
            Tp = int(T * 0.4)
            Tn = int(T * 0.16)
            pulse_trig(pulse, Tp, Tn)
            if derivative:
                pulse = ZeroFilter(pulse)
        elif self.ptype == 'gamma':
            Rg = 1.2
            Rk = 0.3
            Tp = T/(2.0*Rg)
            Te = Tp*(Rk+1.0)
            alpha = 3.0
            beta  = 0.16/(alpha-1)
            pulse_gamma(pulse, Te, alpha, beta)
            if derivative:
                pulse = ZeroFilter(pulse)
        elif self.ptype == 'igamma':
            alpha = 10
            beta  = 0.16*(alpha+1)
            pulse_igamma(pulse, T, alpha, beta)
            if derivative:
                pulse = ZeroFilter(pulse)
        elif self.ptype == 'lf':
            # These three are all in seconds
            Ta = 1.0/(2.0*np.pi*self.Fa)
            Tp = T0/(2.0*self.Rg)
            Te = Tp*(self.Rk+1.0)
            eps = lf_epsilon(Te, Ta, T0)
            alpha = lf_alpha(Tp, Te, eps, T0)
            omega = np.pi / Tp
            pulse_lf(pulse, T0, Te, alpha, omega, eps)
        elif self.ptype == 'invexp':
            for i in range(n):
                t = 1.0-float(i/T)
                pulse[i] = -np.exp(-t*5)
        elif self.ptype == 'polefilter':
            pulse[T/2] = 1.0
            pulse = PoleFilter(pulse, self.pole)
        elif self.ptype == 'zerofilter':
            pulse[T/2] = 1.0
            pulse = ZeroFilter(pulse, self.zero)
        elif self.ptype == 'polezerofilter':
            pulse[T/2] = 1.0
            pulse = PoleFilter(pulse, self.pole)
            pulse = PoleFilter(pulse, self.pole)
            pulse = ZeroFilter(pulse, self.zero)
        elif self.ptype == 'polepairzerofilter':
            pulse[T/2] = 1.0
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

def GFilter(a, mag, angle, pole):
    """
    Takes a magnitude and phase taken to be the location of a pole
    pair; filters the signal using the implied conjugate pair.
    """
    filter = np.zeros(a.size)

    # Memory
    m = [0.0, 0.0]

    # Pole pair
    r1 = 2.0 * mag * np.cos(angle)
    r2 = -mag**2

    peak = 0.0
    opening = True
    for i in range(a.size):
        if not opening:
            if a[i] > 0:
                opening = True
            else:
                filter[i] = a[i] + pole * m[0]
        if opening:
            filter[i] = a[i] + r1 * m[0] +  r2 * m[1]
            if filter[i] < -peak:
                opening = False
        m[1] = m[0]
        m[0] = filter[i]
        if peak < filter[i]:
            peak = filter[i]

    filter *= np.sqrt(len(filter)) / linalg.norm(filter)
    return filter

def PFilter(a, mag, angle, pole):
    filter = a[::-1]
    filter = ssp.PolePairFilter(filter, mag, angle)
    filter = filter[::-1]
    filter = ssp.ZeroFilter(filter, 1.0)
    filter = ssp.PoleFilter(filter, pole)
    return filter
