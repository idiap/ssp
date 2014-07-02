#!/usr/bin/env python
#
# Copyright 2014 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, January 2014
#
import unittest
import ssp
import numpy as np
import numpy.testing as npt

class TestSSP(unittest.TestCase):
    """Tests for SSP package"""

    def setUp(self):
        """
        Generate one (short) frame of a 1 kHz sinusoid at a sampling
        rate of 16 kHz.  It doesn't matter too much what it is, just
        that it is representative of some natural signal.
        """
        self.pcm = ssp.PulseCodeModulation(16000)
        self.seq = np.zeros(64)
        p = self.pcm.seconds_to_period(1.0/1000);
        for s in range(len(self.seq)):
            self.seq[s] = np.sin(2*np.pi * s/p)
        w = ssp.nuttall(len(self.seq)+1)
        w = np.delete(w, -1)
        self.seq = ssp.Window(self.seq, w)

    def testCep(self):
        order = 4
        ac = ssp.Autocorrelation(self.seq)
        ar, g = ssp.ARLevinson(ac, order)
        print "ar: ", ar
        cep = ssp.ARCepstrum(ar, g)
        print "cep:", cep
        ar2, g2 = ssp.ARCepstrumToPoly(cep)
        print "ar: ", ar2
        npt.assert_array_almost_equal(ar, ar2)
        npt.assert_almost_equal(g, g2)

    def testLSP(self):
        order = 4
        ac = ssp.Autocorrelation(self.seq)
        ar, g = ssp.ARLevinson(ac, order)
        print "ar:", ar
        ls = ssp.ARLineSpectra(ar)
        print "ls:", ls
        ar2 = ssp.ARLineSpectraToPoly(ls)
        print "ar:", ar2
        npt.assert_array_almost_equal(ar, ar2)

