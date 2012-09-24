#!/usr/bin/env python2
#
# Copyright 2012 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, July 2012
#
from ssp import *

F0 = 100.0
period = 16000/F0
angle = F0/16000 * 2 * np.pi
order = 18

w = 1024
types = (
    ## ('lf', (1.0, 0.3, 100)),
    ('lf', (1.0, 0.3, 900)),
    ## ('lf', (1.0, 0.3, 400)),
    ## ('lf', (1.3, 0.3, 400)),
    ## ('lf', (1.0, 0.1, 400)),
    ## ('lf', (1.0, 0.3, 400)),
    ## ('impulse', None),
    ## ('invexp', None),
    ('polefilter', (0.97)),
    ('polefilter', (0.9)),
    ('polezerofilter', (0.97, 1.0)),
    ('polezerofilter', (0.95, 1.0)),
    ('polepairzerofilter', (0.95, angle, 1.0)),
    ('polepairzerofilter', (0.93, angle, 1.0))
    )

fig = Figure(len(types), 3)

for pulseType, params in types:
    # Pulse train
    p = pulse(period, pulseType, params)
    p = np.tile(p, int(w/period+1))
    ax1 = fig.subplot()
    ax1.plot(p[:w])

    # Spectrum
    wp = Window(p[:w], np.hanning(w))
    ps = Periodogram(wp)
    ax2 = fig.subplot()
    lp = 10.0 * np.log10(ps)
    ax2.plot(lp - np.max(lp))

    # Filter and residual
    ac = Autocorrelation(p)
    a, g = ARLevinson(ac, order=order)
    ex = ARExcitation(p, a, g)
    ax3 = fig.subplot()
    ax3.plot(ex)

    ax1.set_xlim(0, w)
    ax2.set_xlim(0, w/2+1)
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
