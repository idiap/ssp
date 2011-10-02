#!/usr/bin/python2
#
# Copyright 2011 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner
#
import numpy as np
import matplotlib.pyplot as plt

from ssp import *

#x = BilinearWarpMatrix(3, alpha=-0.3, size=8)
#print x

i = np.identity(30)
o1 = BilinearWarpOppenheim(i, alpha=0, size=40)
o2 = BilinearWarpOppenheim(i, alpha=0.1, size=40)
o3 = BilinearWarpOppenheim(i, alpha=0.3, size=40)
#o5 = BilinearWarpOppenheim(i, alpha=-0.1, size=40)
#o6 = BilinearWarpOppenheim(i, alpha=-0.3, size=40)
o5 = BilinearWarpMatrix(30, alpha=-0.1, size=40)
o6 = BilinearWarpMatrix(30, alpha=-0.3, size=40)


#m = BilinearWarpMatrix(4, alpha=0.42, size=40)

fig = plt.figure()
o1Mat = fig.add_subplot(2,3,1)
o2Mat = fig.add_subplot(2,3,2)
o3Mat = fig.add_subplot(2,3,3)
o5Mat = fig.add_subplot(2,3,5)
o6Mat = fig.add_subplot(2,3,6)

o1Mat.imshow(o1.T)
o2Mat.imshow(o2.T)
o3Mat.imshow(o3.T)
o5Mat.imshow(o5)
o6Mat.imshow(o6)

plt.show()
#print o.T
#print "-----------------"
#print m
