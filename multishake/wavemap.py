#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
import functools

np.seterr(invalid='raise')

"""
The discretisation is made of a grid of “particles”, each with degrees of freedom.
The dimension of the grid is the dimension of the source, the dofs are the dimension
of the space where the target is embedded.

So, for a "chain", i.e., a map (t,x) -> M, the shape is going to be (N,n),
where there are N particles, and the target vector space has dimension n.

Chain examples are: breather, gordon example.

For an evolving surface, i.e., a map (t,x,y) -> M, the shape is (N1,N2,n),
where there are N1 x N2 particles, each with n degrees of freedom.

Surface examples are: Blowup, exact solution.
"""

def local_projection(q, constraint, reaction):
	"""
	Project at a given point in space (so q is a simple vector)
	(not used)
	"""
	local_reaction = reaction(q)
	def residual(dql):
		dq = dql[:len(q)]
		lag = dql[len(q):]
		res1 = dq - np.dot(lag, local_reaction)
		res2 = constraint(q + dq)
		return np.hstack([res1, res2])
	guess = np.zeros(len(q) + len(local_reaction))
	result = nlsolver(residual, guess)
	return result[:len(q)]



def dim_slice(rank, index, content):
	"""
	Compute slice of the form (:,0,:,:,:)
	where `0` is the content at index `index` in the slice.
	"""
	s = [slice(None)] * rank
	s[index] = content
	return tuple(s)

def periodic(QQ, direction):
	"""
	Make a big vector QQ periodic in the given direction
	"""
	rank = np.ndim(QQ)
	QQ[dim_slice(rank, direction, 0)] = QQ[dim_slice(rank, direction, -2)]
	QQ[dim_slice(rank, direction, -1)] = QQ[dim_slice(rank, direction, 1)]


def neumann(QQ, direction):
	"""
	Apply Neumann boundary condition to a scattered vector QQ in one direction
	"""
	rank = np.ndim(QQ)
	QQ[dim_slice(rank, direction, 0)] = QQ[dim_slice(rank, direction, 1)]
	QQ[dim_slice(rank, direction, -1)] = QQ[dim_slice(rank, direction, -2)]

def pad(Q):
	"""
	Add zero on the border, and copy Q in the middle.
	"""
	rank = np.ndim(Q)
	big_shape = list(np.shape(Q))
	for i in range(rank-1):
		big_shape[i] += 2
	QQ = np.zeros(big_shape)
	# copy Q:
	multi_dim_slice = (slice(1,-1),)*(rank-1) + (slice(None),)
	QQ[multi_dim_slice] = Q
	return QQ

def scatter(Q, border):
	"""
	Padding in all directions, with boundary conditions `border`
	"""
	QQ = pad(Q)
	rank = np.ndim(Q)
	# add the extra values on the boundary
	for direction in range(rank-1):
		border(QQ, direction)
	return QQ

def directed_laplace(QQ, direction):
	"""
	Discrete Laplace of a n+m-tensor in given direction (x=0, y=1, ...)
	QQ: padded vector for boundary conditions
	"""
	rank = np.ndim(QQ)
	slicer = functools.partial(dim_slice, rank, direction)
	ddQQ = -2*QQ[slicer(slice(1,-1))] + QQ[slicer(slice(None,-2))] + QQ[slicer(slice(2,None))]
	# remove the extra boundary conditions
	gather = [slice(1,-1)]*(rank-1) + [slice(None)]
	gather[direction] = slice(None)
	ddQ = ddQQ[tuple(gather)]
	return ddQ

def directed_grad(QQ, direction):
	"""
	Gradient of a scattered Q in given direction.
	Works only with Neumann or periodic conditions.
	"""
	dim = np.ndim(QQ)
	slicer = functools.partial(dim_slice, dim, direction)
	dQQ = QQ[slicer(slice(2,None))] - QQ[slicer(slice(1,-1))]
	# remove the extra boundary conditions
	gather = [slice(1,-1)]*(dim-1) + [slice(None)]
	gather[direction] = slice(None)
	dQ = dQQ[tuple(gather)]
	return dQ

class WaveMap(object):
	def __init__(self, dt, dx, border):
		self.dt = dt
		self.dx = dx
		self.border = border

	def directions(self, QQ):
		return range(np.ndim(QQ) - 1)

	def laplace(self, Q):
		"""
		Computes Laplacian
		"""
		Z = np.zeros_like(Q)
		QQ = scatter(Q, self.border)
		for direction in self.directions(Q):
			lap = directed_laplace(QQ, direction)
			Z += lap
		Z /= self.dx**2
		return Z

	def total_force(self, Q):
		return self.laplace(Q)

	def force(self, Q):
		F = self.total_force(Q)
		F *= self.dt*self.dt
		return F

	def elements(self, Q):
		"""
		Volume elements
		"""
		dim = np.ndim(Q) - 1
		return self.dx**dim, self.dx**(dim-2)

	def grad_potential(self, QQ):
		pot = sum(self.directed_grad_potential(QQ, direction) for direction in range(np.ndim(QQ)-1))
		return pot

	def energy(self, Q0, Q1):
		"""
		Compute the energy given the state (Q0,Q1)
		"""
		QQ = scatter(Q0, self.border)
		vol, vol_grad = self.elements(Q0)
		energy = .5*vol*self.kinetic(Q0,Q1)/self.dt/self.dt + .5*vol_grad*self.grad_potential(QQ)
		return energy

	def energy_oscillation(self, Qs):
		"""
		Convenience method: returns all the differences E-E_0, and E_0
		"""
		energy = [self.energy(Qa,Qb) for Qa,Qb in zip(Qs[:-1],Qs[1:])]
		E0 = energy[0]
		denergy = np.array(energy) - E0
		return E0, denergy

	def constraint(self, q):
		return self.scalar_product(q,q) - 1.

	def reaction(self, q):
		"""
		Jacobian of the constraint
		(not used)
		"""
		return np.reshape(q, (1,-1))

	def local_reaction_projection(self, q, q0):
		"""
		Not used.
		Projection using the "reaction force" in the q0 direction
		q0 is assumed to be already normalized
		"""
		product = np.dot(q,q0)
		prod = np.dot(q,q) - 1
		lag = stable_quad(product, prod)
		projected = q + lag*q0
		return projected


	def reaction_projection(self, Q, Q0):
		"""
		Projection using the "reaction force" in the Q0 direction
		Q0 is assumed to be already normalized
		"""
		scalar = self.scalar_product(Q, Q0) # scalar product; half sum of l1 and l2
		product = self.scalar_product(Q, Q) - 1 # product of l1 and l2
		lag_ = - scalar - np.sqrt(scalar**2 - product) # stable version
		lag = product/lag_
		projected = Q + lag[...,np.newaxis]*Q0
		return projected

	def kinetic(self, Q0, Q1):
		"""
		Kinetic energy * dt**2
		"""
		dQ = Q1 - Q0
		kin = np.sum(self.scalar_product(dQ, dQ))
		return kin

	def directed_grad_potential(self, QQ, direction):
		dQ = directed_grad(QQ, direction)
		return np.sum(self.scalar_product(dQ, dQ))

	def normalize(self, Q):
		"""
		Normalisation to length one
		"""
		norm = np.sqrt(self.scalar_product(Q,Q))
		return Q/norm[...,np.newaxis]

	def scalar_product(self, Q1, Q2):
		return np.sum(Q1*Q2, axis=-1)

class HyperbolicWaveMap(WaveMap):
	def scalar_product(self, Q1, Q2):
		signature = -np.ones(Q2.shape[-1])
		signature[-1] = 1.
		return np.sum(Q1*Q2*signature, axis=-1)


def stable_quad(projection, prod):
	"""
	Not used.
	"""
	lag_ = -projection - np.sqrt(projection**2 - prod)
	lag = prod/lag_
	return lag


def get_breather(N, s, j=1, ell=2):
	"""
	Not used.
	"""
	x = np.linspace(0, 2*np.pi, N, endpoint=False) + np.pi/N
	kappa = np.arctan(ell/j*np.tan(j*x))
	ck = np.cos(kappa)
	sk = np.sin(kappa)
	cx = np.cos(ell*x-kappa)
	sx = np.sin(ell*x-kappa)
	cs = np.cos(s/sk)
	ss = np.sin(s/sk)
	u0 = ck*cx - sk*cs*sx
	u1 = ck*sx + sk*cs*cx
	u2 = sk*ss
	u = np.array([u0,u1,u2]).T
	return u

