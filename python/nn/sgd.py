from __future__ import annotations

from typing import List
import numpy as np
import cvmml_api as cvmml


class SGD:
	"""
	Stochastic Gradient Descent optimizer.

	Uses tensor in-place operations (subtract_) to update parameters
	without creating unnecessary intermediate tensors.

	Args:
		parameters : list of Tensor parameters to optimize.
		lr         : learning rate (default: 0.01).
	"""

	def __init__(self, parameters: List, lr: float = 0.01):
		self.parameters = list(parameters)
		self.lr = lr

	def zero_grad(self):
		"""Reset all parameter gradients to zero."""
		for param in self.parameters:
			param.zero_grad()

	def step(self):
		"""
		Apply one SGD update: param -= lr * grad.
		Uses subtract_ (in-place) with tensor * scalar to stay on-device.
		"""
		for param in self.parameters:
			grad = param.grad()
			if grad is None:
				continue
			# Build a gradient tensor with the same shape
			grad_tensor = cvmml.Tensor(list(param.shape()))
			np.asarray(grad_tensor)[:] = grad
			# Scale by learning rate: lr * grad
			scaled = grad_tensor * self.lr
			# In-place update: param -= lr * grad
			if param.device() == cvmml.Device.CUDA:
				scaled = scaled.to_cuda()
			param.subtract_(scaled)
