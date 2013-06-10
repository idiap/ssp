# SSP - Speech Signal Processing module

SSP is released under the Gnu GPL version 3.  See the file `COPYING`
for details.

To install SSP, something like this ought to work:
    mkdir lib
    cd lib
    git clone git@github.com:idiap/ssp.git
    export PYTHONPATH=~/lib

Then in some working directory
    ln -s ~/lib/ssp/bin/pitch.py
    ln -s ~/lib/ssp/bin/codec.py

Then you can say
    pitch.py test.wav  # Graphical view of what's going on
    codec.py -h
    codec.py -r 22050 -p EM1.wav EM1.lf0

Otherwise, see the wiki at https://github.com/idiap/ssp/wiki

--
[Phil Garner](http://www.idiap.ch/~pgarner)  
June 2013
