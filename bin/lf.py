#!/usr/bin/env python2
#
# Copyright 2012 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, July 2012
#
import ssp
import numpy as np
import matplotlib.pyplot as plt

pcm = ssp.PulseCodeModulation(16000)
F0 = 100.0
period = pcm.rate/F0
angle = F0/pcm.rate * 2 * np.pi
order = 18

w = 1024
types = (
    ## ('lf', (1.0, 0.3, 100)),
    #('lf', (1.0, 0.2, 900)),
    ('lf', (1.0, 0.2, 200), None, None),
##('lf', (2.0, 0.2, 200), None, None),
    ##('lf', (3.0, 0.2, 200)),
    ## ('lf', (1.3, 0.3, 400)),
    ## ('lf', (1.0, 0.1, 400)),
    ## ('lf', (1.0, 0.3, 400)),
    ## ('impulse', None),
    ## ('invexp', None),
    ('impulse', None, 'polefilter', (0.95,) ),
    #('polefilter', (0.99)),
    #('polezerofilter', (0.95, 1.0)),
    #('polezerofilter', (0.99, 1.0)),
    #('polepairzerofilter', (0.95, angle, 1.0)),
    ('impulse', None, 'polepairzerofilter', (0.96, angle * 0.5, 1.0)),
    ('impulse', None, 'gfilter', (1.05, angle * 1.5, 0.87)),
    ('impulse', None, 'philspecial', (0.98, angle, 0.8)),
    ('impulse', None, 'philspecial', (0.96, angle, 0.8))
    #('polepairzerofilter', (1.04, angle * 1.4, 1.0))
    #('multipulse', None)
    )

fig = ssp.Figure(len(types), 3)

for pulseType, pulseParams, filterType, filterParams in types:
    # Pulse train
    print pulseType, pulseParams, filterType, filterParams
    gm = ssp.GlottalModel(pulseType, pulseParams)
    p = gm.pulse(period, pcm)
    p = np.tile(p, int(w/period+1))

    # optional filter on top of the pulse
    if filterType == 'polefilter':
        p = ssp.PoleFilter(p, filterParams[0])
    elif filterType == 'zerofilter':
        p = ssp.ZeroFilter(p, filterParams[0])
    elif filterType == 'polezerofilter':
        p = ssp.PoleFilter(p, filterParams[0])
        p = ssp.PoleFilter(p, filterParams[0])
        p = ssp.ZeroFilter(p, filterParams[1])
    elif filterType == 'polepairzerofilter':
        p = ssp.PolePairFilter(p, filterParams[0], filterParams[1])
        p = ssp.ZeroFilter(p, filterParams[2])
    elif filterType == 'gfilter':
        p = ssp.GFilter(p, filterParams[0], filterParams[1], filterParams[2])
    elif filterType == 'philspecial':
        p = ssp.PFilter(p, filterParams[0], filterParams[1], filterParams[2])

    ax1 = fig.subplot()
    ax1.plot(p[:w])

    # Spectrum
    wp = ssp.Window(p[:w], np.hanning(w))
    ps = ssp.Periodogram(wp)
    ax2 = fig.subplot()
    lp = 10.0 * np.log10(ps)
    ax2.plot(lp - np.max(lp))

    # Filter and residual
    ac = ssp.Autocorrelation(p)
    a, g = ssp.ARLevinson(ac, order=order)
    ex = ssp.ARExcitation(p, a, g)
    ax3 = fig.subplot()
    ax3.plot(ex)

    ax1.set_xlim(0, w)
    #ax2.set_xlim(0, w/2+1)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-60, 0)
    ax3.set_xlim(0, w)

# Impulse train
## it = pulse(period, 'impulse')
## it = np.tile(it, w/period+1)
## ax4 = fig.subplot()
## ax4.plot(it[:w])

# Zero filter
## iz1 = ZeroFilter(it, 1.0)
## iz2 = PoleFilter(iz1, -0.4)
## iz2 = PoleFilter(iz2, 0.8)
## ax4.plot(iz1[:w])
## ax4.plot(iz2[:w])

## ax4.set_xlim(0, w)
plt.show()
