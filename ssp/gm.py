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

from . import core
from . import filter

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
    """
    Given Ta, uses Newton-Raphson to find epsilon in an LF model.
    """
    tce = T0 - te
    epsilon = 1.0/ta
    for iter in range(5):
        f = 1.0 - np.exp(-epsilon * tce) - ta * epsilon
        fd = tce * np.exp(-epsilon * tce) - ta
        epsilon -= f/fd

    return epsilon

def lf_te(T0, alpha, omega, epsilon, te=None):
    """
    Given an LF model in terms of alpha, omega and epsilon, calculates
    Te using Newton-Raphson.
    """
    print(alpha, omega,)
    tc = T0
    tp = np.pi / omega
    if te is None:
        # Initialise te to tp, plus a bit to break symmetry
        te = tp * 1.01
    for iter in range(5):
        exp = np.exp(-epsilon*(tc-te))
        sin = np.sin(omega*te)
        tan = np.tan(omega*te)
        esin = np.exp(alpha*te)*sin
        f = (
            alpha
            - omega/tan
            + omega/esin
            - (alpha**2 + omega**2)*((tc-te)*exp/(1.0-exp) - 1.0/epsilon)
            )
        fd = (
            (omega/sin)**2
            - omega/esin*(alpha+omega/tan)
            + (alpha**2+omega**2)
            * (exp*(1-epsilon*(tc-te))-exp**2) / (1-exp)**2
            )
        te -= f/fd
        if te <= tp or te >= tc:
            print("None")
            return None
            # With proper initialisation, this should not happen
            #raise ValueError('te < tp')
    print(te)
    return te

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
                self.Rg = core.parameter('Rg', 1.0)
                self.Rk = core.parameter('Rk', 0.2)
                self.Fa = core.parameter('Fa', 200)
        elif self.ptype == 'zerofilter':
            if params is not None:
                (self.zero) = params
            else:
                self.zero = core.parameter('Zero', 0.97)
        elif self.ptype == 'polefilter':
            if params is not None:
                (self.pole) = params
            else:
                self.pole = core.parameter('Pole', 0.97)
        elif self.ptype == 'polezerofilter':
            if params is not None:
                (self.pole, self.zero) = params
            else:
                self.pole = core.parameter('Pole', 0.97)
                self.zero = core.parameter('Zero', 1.0)
        elif self.ptype == 'polepairzerofilter':
            if params is not None:
                (self.pole, self.angle, self.zero) = params
            else:
                self.pole = core.parameter('Pole', 0.97)
                self.angle = core.parameter('Angle', 0.0)
                self.zero = core.parameter('Zero', 1.0)

    def pulse(self, n, pcm, derivative=True):
        n = int(n)
        pulse = np.zeros((n))
        T0 = pcm.period_to_seconds(n) # Fundamental period in seconds
        T = float(n)
        if self.ptype == 'impulse':
            pulse[int(T/2)] = 1
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
                pulse = core.ZeroFilter(pulse)
        elif self.ptype == 'trig':
            Tp = int(T * 0.4)
            Tn = int(T * 0.16)
            pulse_trig(pulse, Tp, Tn)
            if derivative:
                pulse = core.ZeroFilter(pulse)
        elif self.ptype == 'gamma':
            Rg = 1.2
            Rk = 0.3
            Tp = T/(2.0*Rg)
            Te = Tp*(Rk+1.0)
            alpha = 3.0
            beta  = 0.16/(alpha-1)
            pulse_gamma(pulse, Te, alpha, beta)
            if derivative:
                pulse = core.ZeroFilter(pulse)
        elif self.ptype == 'igamma':
            alpha = 10
            beta  = 0.16*(alpha+1)
            pulse_igamma(pulse, T, alpha, beta)
            if derivative:
                pulse = core.ZeroFilter(pulse)
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
            pulse = core.PoleFilter(pulse, self.pole)
        elif self.ptype == 'zerofilter':
            pulse[T/2] = 1.0
            pulse = core.ZeroFilter(pulse, self.zero)
        elif self.ptype == 'polezerofilter':
            pulse[T/2] = 1.0
            pulse = core.PoleFilter(pulse, self.pole)
            pulse = core.PoleFilter(pulse, self.pole)
            pulse = core.ZeroFilter(pulse, self.zero)
        elif self.ptype == 'polepairzerofilter':
            pulse[T/2] = 1.0
            pulse = core.PolePairFilter(pulse, self.pole, self.angle)
            pulse = core.ZeroFilter(pulse, self.zero)
        elif self.ptype == 'multipulse':
            for i in range(24):
                pulse[i] = 1
        else:
            raise LookupError('Unknown pulse type ' + str(self.ptype))

        # We want the total power in the pulse to average one for each
        # sample, so the power is the length of the pulse
        pulse *= np.sqrt(T) / linalg.norm(pulse)
        return pulse


class IncrementalFilter(filter.Filter):
    """
    Filter, but implemented by hand so we can get the results sample by sample.
    Only does poles for now.
    """
    def __init__(self):
        filter.Filter.__init__(self)
        self.state = None;

    def alloc(self):
        if not self.solved:
            self.solve()
        self.state = np.zeros((len(self.a)))

    def reset(self, start=0):
        if self.state == None:
            return
        for i in range(start, len(self.state)):
            self.state[i] = 0.0

    def stall(self):
        if self.state == None:
            return
        for i in range(1, len(self.state)):
            self.state[i] = self.state[0]

    def filter(self, x):
        if self.state == None:
            self.alloc()
            print("a:", self.a)
        for i in range(1, len(self.state)):
            self.state[-i] = self.state[-i-1]
        self.state[0] = -x
        self.state[0] = -np.dot(self.state, self.a)
        return self.state[0]

class MaxPhaseGlottis:
    def __init__(self):
        self.py = 0.0
        self.max = 0.0
        self.maxphase = IncrementalFilter()
        self.maxphase.addConjugatePole(1, 0) # some default
        self.minphase = IncrementalFilter()
        self.minphase.addConjugatePole(core.parameter("GlottisPole", 0.9))
        #self.minphase.addZero(1)
        self.closure = core.parameter("GlottisClosure", 0.1)

    def setpolepair(self, mag, angle):
        self.maxphase.clear()
        self.maxphase.addConjugatePole(1/mag+1e-8, angle)
        self.maxphase.solve()

    def reset(self):
        self.maxphase.reset()
        self.minphase.reset()

    def glotter(self, e):
        y = np.ndarray((len(e)))
        for i in range(len(e)):
            y[i] = self.maxphase.filter(e[i])
            y[i] = self.minphase.filter(y[i])
            if y[i] > self.max:
                self.max = y[i]
            if y[i] < 0.0:
                y[i] = 0.0
            # Gradient -5 is a bit more than the excitation is likely to yield
            if (y[i]-self.py < -5) and (y[i] < (self.max * self.closure)):
                self.maxphase.reset()
                self.minphase.stall()
            self.py = y[i]
        return y

class MinPhaseGlottis:
    def __init__(self):
        self.py = 0.0
        self.maxphase = filter.Filter()
        self.maxphase.addConjugatePole(1, 0) # some default
        self.minphase = filter.Filter()
        self.minphase.addConjugatePole(core.parameter("GlottisPole", 0.99))
        self.minphase.addZero(1)

    def setpolepair(self, mag, angle):
        self.maxphase.clear()
        self.maxphase.addConjugatePole(mag+1e-8, angle)
        self.maxphase.solve()

    def reset(self):
        self.maxphase.reset()
        self.minphase.reset()

    def glotter(self, e):
        x = self.maxphase.filter(e)
        y = self.minphase.filter(x)
        return y
