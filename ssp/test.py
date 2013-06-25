#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""A few checks
"""

import os, sys
import unittest
import tempfile, shutil

class SSPTest(unittest.TestCase):
  """Performs various tests."""

  def test01_scripts(self):
    """Tests a script"""

    # warp script
    from ssp.script.warp import warp
    parameters = [
      '--no-show', 
    ]
    self.assertEqual(warp(parameters), 0)
