from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import cvmml_api as cvmml

class Optimizer(ABC):
	"""Abstract base class for all optimizers."""
	def __init__(self, parameters: List, lr: float):
		self.parameters = list(parameters)
		self.lr = lr

	def zero_grad(self):
		"""Reset all parameter gradients to zero."""
		for param in self.parameters:
			param.zero_grad()

	@abstractmethod
	def step(self):
		"""Apply one optimization step."""
		pass


class SGD(Optimizer):
	"""
	Stochastic Gradient Descent optimizer.

	Includes support for momentum and weight decay.
	Uses tensor in-place operations (subtract_) to update parameters.
	"""
	def __init__(self, parameters: List, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
		super().__init__(parameters, lr)
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.velocities = []
		for p in self.parameters:
			self.velocities.append(np.zeros(p.size(), dtype=np.float32))

	def step(self):
		for idx, param in enumerate(self.parameters):
			grad = param.grad()
			if grad is None:
				continue

			g = np.array(grad, dtype=np.float32, copy=False)
			if self.weight_decay != 0.0:
				g += self.weight_decay * np.asarray(param).flatten()
			if self.momentum != 0.0:
				self.velocities[idx] = self.momentum * self.velocities[idx] + g
				g = self.velocities[idx]

			grad_tensor = cvmml.Tensor(list(param.shape()))
			np.asarray(grad_tensor)[:] = g.reshape(param.shape())
			scaled = grad_tensor * self.lr
			if param.device() == cvmml.Device.CUDA:
				scaled = scaled.to_cuda()
			param.subtract_(scaled)

class Adam(Optimizer):
	"""
	Adam optimizer.
	
	Implements the Adam algorithm with weight decay.
	"""
	def __init__(self, parameters: List, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
		super().__init__(parameters, lr)
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps
		self.weight_decay = weight_decay
		self.t = 0
		self.m = []
		self.v = []
		for p in self.parameters:
			self.m.append(np.zeros(p.size(), dtype=np.float32))
			self.v.append(np.zeros(p.size(), dtype=np.float32))

	def step(self):
		self.t += 1
		for idx, param in enumerate(self.parameters):
			grad = param.grad()
			if grad is None:
				continue

			g = np.array(grad, dtype=np.float32, copy=False)
			if self.weight_decay != 0.0:
				g += self.weight_decay * np.asarray(param).flatten()

			self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * g
			self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * (g ** 2)

			m_hat = self.m[idx] / (1.0 - (self.beta1 ** self.t))
			v_hat = self.v[idx] / (1.0 - (self.beta2 ** self.t))
			
			update = m_hat / (np.sqrt(v_hat) + self.eps)

			grad_tensor = cvmml.Tensor(list(param.shape()))
			np.asarray(grad_tensor)[:] = update.reshape(param.shape())

			scaled = grad_tensor * self.lr
			if param.device() == cvmml.Device.CUDA:
				scaled = scaled.to_cuda()
			param.subtract_(scaled)
