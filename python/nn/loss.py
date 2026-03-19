from __future__ import annotations

from abc import ABC, abstractmethod
import cvmml_api as cvmml


class LossFunction(ABC):
	"""Abstract base class for all loss functions."""

	@abstractmethod
	def forward(self, predictions, targets):
		pass

	def __call__(self, predictions, targets):
		return self.forward(predictions, targets)


class CrossEntropyLoss(LossFunction):
	"""
	Fused Softmax + Negative Log-Likelihood loss.
	Numerically stable: uses max-shift before exp.

	Expects:
		predictions : Tensor of shape [batch_size, num_classes] (raw logits)
		targets     : Tensor of shape [batch_size, num_classes] (one-hot encoded)

	Returns:
		Scalar Tensor (mean loss over the batch).
	"""

	def forward(self, logits, targets):
		# Numerical stability: shift logits by max per sample
		max_logits = logits.max([1], True)
		shifted = logits - max_logits

		# log_softmax = shifted - log(sum(exp(shifted)))
		exp_shifted = shifted.exp()
		sum_exp = exp_shifted.sum([1], True)
		log_sum_exp = sum_exp.log()
		log_softmax = shifted - log_sum_exp

		# NLL: -sum(targets * log_softmax) / batch_size
		nll = (targets * log_softmax).sum([1], False)
		loss = nll.sum([0], False) * (-1.0 / logits.shape()[0])
		return loss
