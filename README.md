# SSP - Speech Signal Processing module

## Overview

SSP is a package for doing signal processing in python; the functionality is
biassed towards speech signals.  Top level programs include a feature extracter
for speech recognition, and a vocoder for both coding and speech synthesis.
The vocoder is based on linear prediction, but with several experimental
excitation models.  A continuous pitch extraction algorithm is also provided,
built around standard components and a Kalman filter.

There is a "sister" package, [libssp](https://github.com/idiap/libssp), that
includes translations of some algorithms in C++.  Libssp is built around
[libube](https://github.com/pgarner/libube) that makes this translation easier.

SSP is released under a BSD licence.  See the file `COPYING` for details.

## Installation

To install SSP from git, just clone it into a working directory:
```sh
cd ~/src  # Just for example
git clone git@github.com:idiap/ssp.git
cd ssp
```
Or, if you work in a different directory:
```sh
cd where-I-work
ln -s ~/src/ssp/pitch.py
ln -s ~/src/ssp/codec.py
```
i.e., you shouldn't need to set `PYTHONPATH`

Then you can say
```sh
pitch.py test.wav  # Graphical view of what's going on
codec.py -h
codec.py -r 22050 -p EM1.wav EM1.lf0
```

If you know what buildout is:
```sh
python bootstrap.py
./bin/buildout
```

Alternatively, SSP is available on
[PyPI](https://pypi.python.org/pypi) at
https://pypi.python.org/pypi/ssp  That's available by typing
```sh
pip install ssp  # Root level installation
```
or
```sh
pip install ssp --user  # User level in ~/.local
```
Note that the PyPI package is very likely to be out of date!

Otherwise, see the wiki at https://github.com/idiap/ssp/wiki

## Publications

The pitch tracker in SSP is documented in the paper:
```
@Article{Garner2013,
  author =       "Garner, Philip N. and Cernak, Milos and Motlicek,
                  Petr",
  title =        "A Simple Continuous Pitch Estimation Algorithm",
  journal =      "IEEE Signal Processing Letters",
  year =         2013,
  month =        "January",
  volume =       20,
  number =       1,
  pages =        "102--105",
  doi =          "10.1109/LSP.2012.2231675"
}
```
and there is a [downloadable
pdf](http://publications.idiap.ch/downloads/papers/2012/Garner_SPL_2012.pdf).

The codec is documented in a technical report:
```
@TechReport{GarnerTech2015,
  author =      "Garner, Philip N. and Cernak, Milos and Potard, Blaise",
  title =       "A simple continuous excitation model for parametric vocoding",
  institution = "Idiap Research Institute",
  year =        2015,
  type =        "Idiap-RR",
  number =	    "03-2015",
  month =       "January"
}
```
and again there is a [downloadable pdf](http://publications.idiap.ch/downloads/reports/2014/Garner_Idiap-RR-03-2015.pdf).

--
[Phil Garner](http://www.idiap.ch/~pgarner)
June 2013
