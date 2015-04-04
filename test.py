#!/usr/bin/python
#
# Copyright 2014 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, January 2014
#
import unittest

from ssp.test import TestSSP

testCase = TestSSP('testLSP')
unittest.TextTestRunner().run(testCase)
