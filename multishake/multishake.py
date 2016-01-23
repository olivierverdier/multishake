#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

# q: 4 tensor, t,x,y,q

def leap_frog(Q0, Q1, force):
	"""
	The standard leap frog method
	"""
	return 2*Q1 - Q0 + force(Q1)

def global_projection(Q, reaction_projection, Q0):
	return reaction_projection(Q, Q0)

def shake(Q0, Q1, force, reaction_projection):
	"""
	One step of Shake
	Time step is encoded in force
	"""
	Q2 = leap_frog(Q0, Q1, force)
	return global_projection(Q2, reaction_projection, Q1)


