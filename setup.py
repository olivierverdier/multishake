#!/usr/bin/env python

from distutils.core import setup


setup(
	name         = 'multishake',
	description  = 'Multi-symplectic solver in Python',
	author = 'David Cohen, Olivier Verdier',
	url = 'https://github.com/olivierverdier/multishake',
	license      = 'GPL v.3',
	keywords = ['Math', 'multi-symplectic', 'wave map',],
	packages=['multishake',],
	classifiers = [
	'Development Status :: 4 - Beta',
	'Intended Audience :: Science/Research',
	'License :: OSI Approved :: BSD License',
	'Operating System :: OS Independent',
	'Programming Language :: Python',
	'Topic :: Scientific/Engineering :: Mathematics',
	],
	)
