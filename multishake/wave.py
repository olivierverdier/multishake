#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

def get_plane_wave(wave_vector, amplitude, phase):
	"""
	Return a plane wave with wave vector (k1,k2), both integers.
	"""
	wave_length = np.sqrt(np.sum(np.square(wave_vector)))
	def wave(t,x,y):
		return amplitude * np.cos(2*np.pi*(wave_vector[0]*x + wave_vector[1]*y - wave_length * t - phase))
	return wave

def get_wave_initial(wave_vector, amplitude, phase):
	"""
	Return initial condition corresponding to the wave.
	"""
	wave = get_plane_wave(wave_vector, amplitude, phase)
	return functools.partial(wave, 0.)

def compute_wave(wave_vector, amplitude, phase, t, x, y):
	"""
	Compute the result of the wave with corresponding characteristics at t,x,y
	"""
	wave_length = np.sqrt(np.sum(np.square(wave_vector)))
	return amplitude * np.cos(2*np.pi*(wave_vector[0]*x + wave_vector[1]*y - wave_length * t - phase))

class Wave2D(object):
	"""
	Compute a sum of plane waves in 2D.
	"""
	def __init__(self, waves):
		"""
		The list waves should consist of elements of type (wave_vector, amplitude, phase)
		The wave_vector is an array of length two.
		"""
		self.waves = list(waves)

	def __call__(self, t, x, y):
		return np.sum(np.array([compute_wave(wave_vector,amplitude,phase,t,x,y) for (wave_vector, amplitude, phase) in self.waves]), axis=0)


