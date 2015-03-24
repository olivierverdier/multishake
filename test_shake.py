#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as npt

import wavemap, multishake
import numpy as np

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

class TestDims(unittest.TestCase):
	def test_slice(self):
		computed = wavemap.dim_slice(5,2,10)
		expected = (slice(None), slice(None), 10, slice(None), slice(None))

	def setUp(self):
		n = 3
		m = 3
		p = 1
		self.Q = np.arange(m*n*p).reshape((m,n,p))

	def test_scatter_per(self):
		Q = self.Q
		QQ = wavemap.scatter(Q, wavemap.periodic)
		i,j = 1,2
		npt.assert_allclose(QQ[i+1,j+1], Q[i,j])
		npt.assert_allclose(QQ[0]-QQ[-2],0)
		npt.assert_allclose(QQ[:,1]-QQ[:,-1],0)

	def test_scatter_neu(self):
		Q = self.Q
		QQ = wavemap.scatter(Q, wavemap.neumann)
		i,j = 1,2
		npt.assert_allclose(QQ[i+1,j+1], Q[i,j])
		npt.assert_allclose(QQ[0,:]-QQ[1,:],0)
		npt.assert_allclose(QQ[:,-2]-QQ[:,-1],0)



	def test_laplace(self):
		Q = self.Q
		QQ = wavemap.scatter(Q, wavemap.periodic)
		L = wavemap.directed_laplace(QQ, 0)
		self.assertEqual(np.shape(L), np.shape(Q))

	def test_dir_grad(self):
		Q = np.ones([5,5,3])
		QQ = wavemap.scatter(Q, wavemap.periodic)
		dQ = wavemap.directed_grad(QQ, 0)
		self.assertEqual(np.shape(dQ), (5,5,3))
		npt.assert_allclose(dQ, 0.)

	def test_grad_pot(self):
		Q = np.ones([5,5,3])
		QQ = wavemap.scatter(Q, border=wavemap.periodic)
		pot = wavemap.grad_potential(QQ)
		npt.assert_allclose(pot, 0.)

	def test_kinetic(self):
		Q0 = np.ones([5,5,4])
		Q1 = np.ones([5,5,4])
		wm = get_wavemap()
		wm.energy(Q0,Q1)

	def test_energy_2D(self):
		bound = .5
		def generate():
			for N in [90,120,200, 400]:
				X, Y = np.ogrid[-bound:bound:N*1j,-bound:bound:N*1j]
				#
				## Q = wavemap.get_equivariant(X, Y, wavemap.quartic_spike)
				#
				rr = np.square(X) + np.square(Y)
				r = np.sqrt(rr)
				pQ = wavemap.quartic_spike(r)
				#
				## pQ = np.cos(2*np.pi*X) + 0*Y
				Q = pQ[:,:,np.newaxis]
				npt.assert_allclose(wavemap.kinetic(Q,Q), 0.)
				## print 'q0', wavemap.directed_grad_potential(Q, 0)
				## print 'g0', wavemap.directed_grad_potential(wavemap.scatter(Q,wavemap.neumann), 0)
				## print 'g1', wavemap.directed_grad_potential(wavemap.scatter(Q,wavemap.periodic), 1)
				wm = get_wavemap(dt=.1, dx=1./N, border=wavemap.neumann)
				## print 'E', wm.energy(Q, Q)
				yield wm.energy(Q, Q)
		es = np.array(list(generate()))
		npt.assert_allclose(es, es[0], rtol=1e-2)



	def test_energy_1D(self):
		latitude = .5
		def generate():
			for N in [40,100,200]:
				thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
				X0 = np.array([np.cos(thetas), np.sin(thetas), latitude*np.ones_like(thetas)]).T
				wm = get_wavemap(dt=.1, dx=1./N)
				Q = global_projection(X0, wm.reaction_projection, X0)
				yield wm.energy(Q, Q)
		es = np.array(list(generate()))
		npt.assert_allclose(es, es[0], rtol=1e-2)

	def test_grad_hat(self):
		bound = .5
		N = 20
		X, Y = np.ogrid[-bound:bound:N*1j,-bound:bound:N*1j]


def get_wavemap(dt=.1, dx=.1, border=wavemap.periodic):
	wm = wavemap.WaveMap(dt=dt, dx=dx, border=border)
	return wm

def get_shake_stepper(dt=.1, dx=.1):
	wm = get_wavemap(dt,dx)
	def step(Q0, Q1):
		return multishake.shake(Q0, Q1, force=wm.force, reaction_projection=wm.reaction_projection)
	return step


class TestZero(unittest.TestCase):
	def test_constant(self):
		"""
		No testing here...
		"""
		n = 3
		p = 2
		Q0 = np.ones([n,n,p])
		Q1 = np.ones([n,n,p])
		step = get_shake_stepper()
		Q2 = step(Q0, Q1)
		## print Q2

class TestProjection(unittest.TestCase):
	def setUp(self):
		self.wm = get_wavemap()

	def test_normalization(self):
		q = np.ones([2])
		wm = self.wm
		proj = wm.normalize(q)
		npt.assert_allclose(proj[0]**2 + proj[1]**2, 1.)

	@unittest.skip("Reaction projection expects that Q0 is normalized; not the case here")
	def test_global_projection(self):
		n,m,p = 3,4,3
		shape = (n,m,p)
		Q = np.ones(shape)
		Qp = global_projection(Q, self.wm.reaction_projection, Q)
		for it in space_iterator(shape):
			npt.assert_allclose(np.sum(np.square(Qp[it])), 1.)

	def notest_reaction_projection(self):
		"""
		This test is not precise enough
		"""
		q = np.array([1.,.3])
		q0 = np.array([1.,0])
		wm = self.wm
		p = wm.reaction_projection(q, q0)
		new_norm = p[0]**2 + p[1]**2
		npt.assert_allclose(new_norm, 1.)

	def test_iterator(self):
		n,m,p = 3,4,4
		iterator = space_iterator((n,m,p))
		self.assertEqual(len(list(iterator)), n*m)

	def test_vector_reaction_projection(self):
		n,m,p = 3,3,4
		Q0 = np.ones([n,m,p])
		Q0[:,:,1:] = 0.
		Q0_lengths = np.sum(Q0*Q0, axis=-1)
		npt.assert_allclose(Q0_lengths, 1.)
		Q1 = .4*np.ones([n,m,p])
		Q2 = self.wm.reaction_projection(Q1,Q0)
		lengths = np.sum(Q2*Q2, axis=-1)
		npt.assert_allclose(lengths, 1.)


class TestEquiInitial(unittest.TestCase):
	def setUp(self):
		self.N = 5
		self.X, self.Y = np.ogrid[-.5:.5:self.N*1j, -.5:.5:self.N*1j]

	def test_quarticspike(self):
		rr = np.square(self.X) + np.square(self.Y)
		r = np.sqrt(rr)
		res = wavemap.quartic_spike(r)
		npt.assert_allclose(res[0,0],0.)
		npt.assert_allclose(res[0,self.N//2], 0.)
		npt.assert_allclose(res[self.N//2, self.N//2],1.)

	def test_equi(self):
		equi = wavemap.get_equivariant(self.X, self.Y, wavemap.quartic_spike)
		self.assertEqual(np.shape(equi), (self.N, self.N, 3))
		npt.assert_allclose(equi[0,0], np.array([0.,0,-1]), err_msg='south pole at the boundaries')
		mid = self.N//2
		npt.assert_allclose(equi[mid,mid], np.array([0.,0,1]), err_msg='north pole at the middle')
		x = mid
		y = mid + mid//2
		npt.assert_allclose(equi[x,y,0]*(-self.Y[0,y]) + equi[x,y,1]*self.X[x,0], 0.)
