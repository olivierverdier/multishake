#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

# q: 4 tensor, t,x,y,q
import scipy.optimize
nlsolver = scipy.optimize.fsolve

def leap_frog(Q0, Q1, force):
	"""
	The standard leap frog method
	"""
	return 2*Q1 - Q0 + force(Q1)

def space_iterator(shape):
	"""
	Iterate over the space components of Q
	"""
	space_shape = shape[:-1]
	return np.ndindex(*space_shape)

def global_projection(Q, reaction_projection, Q0):
	"""
	Projection on constraint using the given local projection function
	"""
	Qp = np.zeros_like(Q)
	for index in space_iterator(np.shape(Q)):
		projected = reaction_projection(Q[index], Q0[index])
		Qp[index] = projected
	return Qp

def shake(Q0, Q1, force, reaction_projection):
	"""
	One step of Shake
	Time step is encoded in force
	"""
	Q2 = leap_frog(Q0, Q1, force)
	return global_projection(Q2, reaction_projection, Q1)


