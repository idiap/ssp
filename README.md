# SSP - Speech Signal Processing module

SSP is released under the Gnu GPL version 3.  See the file `COPYING`
for details.

To install SSP from git, something like this ought to work:
```sh
cd ~/src  # Just for example
git clone git@github.com:idiap/ssp.git
cd ssp
python bootstrap.py
./bin/buildout
```
Then in some working directory
```sh
ln -s ~/src/ssp/bin/pitch.py
ln -s ~/src/ssp/bin/codec.py
```

Then you can say
```sh
pitch.py test.wav  # Graphical view of what's going on
codec.py -h
codec.py -r 22050 -p EM1.wav EM1.lf0
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

Otherwise, see the wiki at https://github.com/idiap/ssp/wiki

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
  pages =        "102--105"
}
```
and there is a [downloadable
pdf](http://publications.idiap.ch/downloads/papers/2012/Garner_SPL_2012.pdf).

--
[Phil Garner](http://www.idiap.ch/~pgarner)  
June 2013
