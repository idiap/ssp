SSP - Speech Signal Processing module

December 2012; converted to a module, so:

 ./core.py is the core library code (was ssp.py)
 ./bin/* is all the executables (make a symlink if you like)

The directory *above* this should be on PYTHONPATH, then
 import ssp
will actually import
 ./__init__.py
and that one will import core.py into namespace ssp

--
Phil Garner
