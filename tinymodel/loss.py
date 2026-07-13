"""Loss functions for TinyMML models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LossFunction(ABC):
    """Callable loss-function protocol."""

    @abstractmethod
    def forward(self, predictions, targets):
        """Return a scalar loss tensor."""

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)


class CrossEntropyLoss(LossFunction):
    """Stable mean cross-entropy for logits and one-hot targets."""

    def forward(self, logits, targets):
        if len(logits.shape()) != 2 or targets.shape() != logits.shape():
            raise ValueError(
                "CrossEntropyLoss expects logits and one-hot targets with shape [B, C]"
            )
        shifted = logits - logits.max([1], True)
        log_softmax = shifted - shifted.exp().sum([1], True).log()
        negative_log_likelihood = (targets * log_softmax).sum([1], False)
        return negative_log_likelihood.sum([0], False) * (-1.0 / logits.shape()[0])


class MSELoss(LossFunction):
    """Mean squared error over every tensor dimension."""

    def forward(self, predictions, targets):
        if predictions.shape() != targets.shape():
            raise ValueError("MSELoss predictions and targets must have identical shapes")
        difference = predictions - targets
        squared_error = difference * difference
        return squared_error.mean(list(range(len(squared_error.shape()))), False)
