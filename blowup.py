#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

def get_equivariant(X, Y, profile):
	"""
	An equivariant initial condition corresponding to the profile a(r).
	"""
	rr = np.square(X) + np.square(Y)
	r = np.sqrt(rr)
	prof = profile(r)
	proff = prof * prof
	tensor = np.array([
	2*X*prof,
	2*Y*prof,
	proff - rr,
	])
	normed = tensor/(proff + rr)
	# we need to put the local dofs last in the array:
	res = np.transpose(normed, [1,2,0])
	return res

def quartic_spike(r):
	"""
	Vectorial function returning profile (1-2r)**4 if r < .5, zero otherwise.
	"""
	non_zero_indices = r < .5
	res = np.zeros_like(r)
	res[non_zero_indices] = np.power(1 - 2*r[non_zero_indices], 4)
	return res
