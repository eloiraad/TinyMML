"""Gradient-based optimizers for TinyMML parameters."""

from abc import ABC, abstractmethod

import numpy as np
import tinytensor as tt


def _tensor_from(values: np.ndarray, device) -> tt.Tensor:
    """Copy one float32 array into a tensor on the requested device."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    tensor = tt.Tensor(list(values.shape))
    np.asarray(tensor)[:] = values
    return tensor.to_cuda() if device == tt.Device.CUDA else tensor


class Optimizer(ABC):
    """Base optimizer holding trainable parameters and learning rate."""

    def __init__(self, parameters: list, lr: float) -> None:
        if lr <= 0.0:
            raise ValueError("learning rate must be positive")
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    @abstractmethod
    def step(self) -> None:
        """Update every parameter with an available gradient."""


class SGD(Optimizer):
    """Stochastic gradient descent with momentum and L2 weight decay."""

    def __init__(
        self,
        parameters: list,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [
            np.zeros(parameter.shape(), dtype=np.float32)
            for parameter in self.parameters
        ]

    def step(self) -> None:
        for index, parameter in enumerate(self.parameters):
            gradient_buffer = parameter.grad()
            if gradient_buffer is None:
                continue
            gradient = np.array(gradient_buffer, dtype=np.float32, copy=True)
            if self.weight_decay:
                gradient += self.weight_decay * np.asarray(parameter)
            if self.momentum:
                self.velocities[index] = (
                    self.momentum * self.velocities[index] + gradient
                )
                gradient = self.velocities[index]
            update = _tensor_from(gradient * self.lr, parameter.device())
            parameter.subtract_(update)


class Adam(Optimizer):
    """Adam optimizer with bias correction and L2 weight decay."""

    def __init__(
        self,
        parameters: list,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if eps <= 0.0 or weight_decay < 0.0:
            raise ValueError("eps must be positive and weight_decay non-negative")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [
            np.zeros(parameter.shape(), dtype=np.float32)
            for parameter in self.parameters
        ]
        self.v = [
            np.zeros(parameter.shape(), dtype=np.float32)
            for parameter in self.parameters
        ]

    def step(self) -> None:
        self.t += 1
        for index, parameter in enumerate(self.parameters):
            gradient_buffer = parameter.grad()
            if gradient_buffer is None:
                continue
            gradient = np.array(gradient_buffer, dtype=np.float32, copy=True)
            if self.weight_decay:
                gradient += self.weight_decay * np.asarray(parameter)

            self.m[index] = (
                self.beta1 * self.m[index] + (1.0 - self.beta1) * gradient
            )
            self.v[index] = (
                self.beta2 * self.v[index]
                + (1.0 - self.beta2) * gradient * gradient
            )
            first_moment = self.m[index] / (1.0 - self.beta1**self.t)
            second_moment = self.v[index] / (1.0 - self.beta2**self.t)
            update_values = first_moment / (np.sqrt(second_moment) + self.eps)
            update = _tensor_from(update_values * self.lr, parameter.device())
            parameter.subtract_(update)
