#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest

from wavemap import get_breather

class TestBreather(unittest.TestCase):
	def test_init(self):
		N = 32
		dt = 1/N/2.
		Q0 = get_breather(N, 0., j=1, ell=2)
		Q1 = get_breather(N, dt, j=1, ell=2)

