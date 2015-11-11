#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy as np
import numpy.testing as npt

import wavemap
import blowup
from utils import get_wavemap

class TestEquiInitial(unittest.TestCase):
	def setUp(self):
		self.N = 5
		self.X, self.Y = np.ogrid[-.5:.5:self.N*1j, -.5:.5:self.N*1j]

	def test_quarticspike(self):
		rr = np.square(self.X) + np.square(self.Y)
		r = np.sqrt(rr)
		res = blowup.quartic_spike(r)
		npt.assert_allclose(res[0,0],0.)
		npt.assert_allclose(res[0,self.N//2], 0.)
		npt.assert_allclose(res[self.N//2, self.N//2],1.)

	def test_equi(self):
		equi = blowup.get_equivariant(self.X, self.Y, blowup.quartic_spike)
		self.assertEqual(np.shape(equi), (self.N, self.N, 3))
		npt.assert_allclose(equi[0,0], np.array([0.,0,-1]), err_msg='south pole at the boundaries')
		mid = self.N//2
		npt.assert_allclose(equi[mid,mid], np.array([0.,0,1]), err_msg='north pole at the middle')
		x = mid
		y = mid + mid//2
		npt.assert_allclose(equi[x,y,0]*(-self.Y[0,y]) + equi[x,y,1]*self.X[x,0], 0.)


	def test_energy_2D(self):
		bound = .5
		def generate():
			for N in [90,120,200, 400]:
				X, Y = np.ogrid[-bound:bound:N*1j,-bound:bound:N*1j]
				#
				## Q = blowup.get_equivariant(X, Y, blowup.quartic_spike)
				#
				rr = np.square(X) + np.square(Y)
				r = np.sqrt(rr)
				pQ = blowup.quartic_spike(r)
				#
				## pQ = np.cos(2*np.pi*X) + 0*Y
				Q = pQ[:,:,np.newaxis]
				wm = get_wavemap()
				npt.assert_allclose(wm.kinetic(Q,Q), 0.)
				## print 'q0', wavemap.directed_grad_potential(Q, 0)
				## print 'g0', wavemap.directed_grad_potential(wavemap.scatter(Q,wavemap.neumann), 0)
				## print 'g1', wavemap.directed_grad_potential(wavemap.scatter(Q,wavemap.periodic), 1)
				wm = get_wavemap(dt=.1, dx=1./N, border=wavemap.neumann)
				## print 'E', wm.energy(Q, Q)
				yield wm.energy(Q, Q)
		es = np.array(list(generate()))
		npt.assert_allclose(es, es[0], rtol=1e-2)



