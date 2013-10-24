#!/usr/bin/python
#
# Copyright 2013 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Phil Garner, October 2013
#     based on a Bob version by Andre Anjos
#

from setuptools import setup, find_packages

setup(
    name='ssp',
    version='0.2',
    description='A small collection of Speech Signal Processing functions',

    url='http://github.com/idiap/ssp',
    license='GPLv3',
    author='Phil Garner',
    author_email='Phil.Garner@idiap.ch',
    long_description=open('README.md').read(),
    packages=find_packages(),
    include_package_data=True,

    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'matplotlib',
        'nose',
        ],

    scripts = [
        'codec.py',
        'extracter.py',
        'spectrogram.py',
    ],

    test_suite='nose.main',

    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
