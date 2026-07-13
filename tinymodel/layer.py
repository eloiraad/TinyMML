"""Neural-network layers built on the TinyTensor extension."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import numpy as np
import tinytensor as tt


class Layer(ABC):
    """Base protocol for buildable, trainable layers."""

    def __init__(self) -> None:
        self._training = True
        self._built = False
        self._input_shape = None
        self._output_shape = None

    @property
    def training(self) -> bool:
        return self._training

    @property
    def built(self) -> bool:
        return self._built

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def train(self) -> "Layer":
        self._training = True
        return self

    def eval(self) -> "Layer":
        self._training = False
        return self

    def build(self, input_shape):
        if self._built:
            return self._output_shape
        self._input_shape = list(input_shape) if input_shape is not None else None
        output_shape = self._build(input_shape)
        self._output_shape = list(output_shape) if output_shape is not None else None
        self._built = True
        return self._output_shape

    def __call__(self, inputs):
        if not self._built:
            inferred_shape = inputs.shape() if hasattr(inputs, "shape") else None
            self.build(inferred_shape)
        return self.forward(inputs)

    def parameters(self) -> list:
        return []

    @abstractmethod
    def _build(self, input_shape):
        """Create parameters and return output shape."""

    @abstractmethod
    def forward(self, inputs):
        """Transform one tensor."""


class Linear(Layer):
    """Apply ``inputs @ weight + bias`` on the final dimension."""

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        init: str = "xavier",
    ) -> None:
        super().__init__()
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if init not in {"xavier", "he"}:
            raise ValueError("init must be 'xavier' or 'he'")
        self.out_features = out_features
        self.use_bias = bias
        self.init = init
        self.weight = None
        self.bias = None

    def _build(self, input_shape):
        if input_shape is None or len(input_shape) < 2:
            raise ValueError("Linear expects input shape [..., in_features]")
        input_features = input_shape[-1]
        scale = math.sqrt((2.0 if self.init == "he" else 1.0) / input_features)
        self.weight = tt.Tensor.randn(
            [input_features, self.out_features],
            0.0,
            1.0,
        ) * scale
        self.weight.set_requires_grad(True)
        if self.use_bias:
            self.bias = tt.Tensor.zeros([1, self.out_features])
            self.bias.set_requires_grad(True)
        output_shape = list(input_shape)
        output_shape[-1] = self.out_features
        return output_shape

    def forward(self, inputs):
        outputs = inputs.mm(self.weight)
        return outputs if self.bias is None else outputs + self.bias

    def parameters(self) -> list:
        return [self.weight] if self.bias is None else [self.weight, self.bias]


class Softmax(Layer):
    """Apply numerically stable softmax along one axis."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def _build(self, input_shape):
        if input_shape is None:
            raise ValueError("Softmax requires an input shape")
        rank = len(input_shape)
        axis = self.axis if self.axis >= 0 else self.axis + rank
        if axis < 0 or axis >= rank:
            raise ValueError("Softmax axis is out of range")
        self.axis = axis
        return input_shape

    def forward(self, inputs):
        shifted = inputs - inputs.max([self.axis], True)
        exponentials = shifted.exp()
        return exponentials / exponentials.sum([self.axis], True)


class ReLU(Layer):
    """Apply rectified linear activation elementwise."""

    def _build(self, input_shape):
        return input_shape

    def forward(self, inputs):
        return inputs.relu()


class Flatten(Layer):
    """Flatten every non-batch dimension into one feature dimension."""

    def _build(self, input_shape):
        if input_shape is None or len(input_shape) < 2:
            raise ValueError("Flatten expects input shape [batch, ...]")
        return [input_shape[0], math.prod(input_shape[1:])]

    def forward(self, inputs):
        return inputs.contiguous().view([inputs.shape()[0], -1])


class Dropout(Layer):
    """Apply inverted dropout during training."""

    def __init__(self, rate: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = rate

    def _build(self, input_shape):
        return input_shape

    def forward(self, inputs):
        if not self.training or self.rate == 0.0:
            return inputs
        keep_probability = 1.0 - self.rate
        mask = tt.Tensor.bernoulli(list(inputs.shape()), keep_probability)
        if inputs.device() == tt.Device.CUDA:
            mask = mask.to_cuda()
        return inputs * mask * (1.0 / keep_probability)


class Conv2D(Layer):
    """Apply an NCHW 2D convolution using ``im2col`` and matrix multiply."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        init: str = "xavier",
    ) -> None:
        super().__init__()
        if out_channels <= 0 or kernel_size <= 0 or stride <= 0 or padding < 0:
            raise ValueError("Conv2D dimensions must be positive and padding non-negative")
        if init not in {"xavier", "he"}:
            raise ValueError("init must be 'xavier' or 'he'")
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.init = init
        self.weight = None
        self.bias = None

    def _build(self, input_shape):
        if input_shape is None or len(input_shape) != 4:
            raise ValueError("Conv2D expects input shape [B, C_in, H, W]")
        input_channels, height, width = input_shape[1:]
        output_height = (
            height + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        output_width = (
            width + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Conv2D kernel, stride, and padding yield an empty output")

        fan_in = input_channels * self.kernel_size * self.kernel_size
        scale = math.sqrt((2.0 if self.init == "he" else 1.0) / fan_in)
        self.weight = tt.Tensor.randn(
            [self.out_channels, fan_in],
            0.0,
            1.0,
        ) * scale
        self.weight.set_requires_grad(True)
        if self.use_bias:
            self.bias = tt.Tensor.zeros([1, self.out_channels, 1, 1])
            self.bias.set_requires_grad(True)
        return [input_shape[0], self.out_channels, output_height, output_width]

    def forward(self, inputs):
        batch_size, _, height, width = inputs.shape()
        output_height = (
            height + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        output_width = (
            width + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        columns = inputs.im2col(
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
        )
        columns = columns.T(1, 2).contiguous()
        weights = self.weight.T(0, 1).contiguous()
        outputs = columns.mm(weights).T(1, 2).contiguous()
        outputs = outputs.view(
            [batch_size, self.out_channels, output_height, output_width]
        )
        return outputs if self.bias is None else outputs + self.bias

    def parameters(self) -> list:
        return [self.weight] if self.bias is None else [self.weight, self.bias]


class BatchNorm(Layer):
    """Normalize each feature channel using batch statistics."""

    def __init__(self, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self._axes = None

    def _build(self, input_shape):
        if input_shape is None or len(input_shape) < 2:
            raise ValueError("BatchNorm expects input shape [B, features, ...]")
        parameter_shape = [1, input_shape[1], *([1] * (len(input_shape) - 2))]
        self._axes = [0, *range(2, len(input_shape))]
        self.gamma = tt.Tensor.ones(parameter_shape)
        self.beta = tt.Tensor.zeros(parameter_shape)
        self.gamma.set_requires_grad(True)
        self.beta.set_requires_grad(True)
        self.running_mean = tt.Tensor.zeros(parameter_shape)
        self.running_var = tt.Tensor.ones(parameter_shape)
        return input_shape

    def forward(self, inputs):
        if self.training:
            mean = inputs.mean(self._axes, True)
            variance = inputs.var(self._axes, True)
            momentum = self.momentum
            running_mean = np.asarray(self.running_mean)
            running_variance = np.asarray(self.running_var)
            running_mean[:] = (
                (1.0 - momentum) * running_mean
                + momentum * np.asarray(mean.contiguous())
            )
            running_variance[:] = (
                (1.0 - momentum) * running_variance
                + momentum * np.asarray(variance.contiguous())
            )
        else:
            mean = self.running_mean
            variance = self.running_var
        normalized = (inputs - mean) / (variance + self.eps).sqrt()
        return self.gamma * normalized + self.beta

    def parameters(self) -> list:
        return [self.gamma, self.beta]


class LayerNorm(Layer):
    """Normalize each sample across its final dimension."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = None
        self.beta = None

    def _build(self, input_shape):
        if input_shape is None or len(input_shape) < 1:
            raise ValueError("LayerNorm requires at least one input dimension")
        parameter_shape = [1, input_shape[-1]]
        self.gamma = tt.Tensor.ones(parameter_shape)
        self.beta = tt.Tensor.zeros(parameter_shape)
        self.gamma.set_requires_grad(True)
        self.beta.set_requires_grad(True)
        return input_shape

    def forward(self, inputs):
        last_axis = len(inputs.shape()) - 1
        mean = inputs.mean([last_axis], True)
        variance = inputs.var([last_axis], True)
        normalized = (inputs - mean) / (variance + self.eps).sqrt()
        return self.gamma * normalized + self.beta

    def parameters(self) -> list:
        return [self.gamma, self.beta]


class ResidualBlock(Layer):
    """Apply child layers and add their result to the original input."""

    def __init__(self, *layers: Layer) -> None:
        super().__init__()
        self.sub_layers = list(layers)

    def _build(self, input_shape):
        current_shape = list(input_shape) if input_shape is not None else None
        for layer in self.sub_layers:
            current_shape = (
                layer.build(current_shape) if not layer.built else layer.output_shape
            )
        if current_shape != list(input_shape):
            raise ValueError(
                "ResidualBlock layers must preserve input shape: "
                f"received {input_shape}, produced {current_shape}"
            )
        return current_shape

    def forward(self, inputs):
        outputs = inputs
        for layer in self.sub_layers:
            outputs = layer(outputs)
        return inputs + outputs

    def parameters(self) -> list:
        parameters = []
        for layer in self.sub_layers:
            parameters.extend(layer.parameters())
        return parameters

    def train(self) -> "ResidualBlock":
        super().train()
        for layer in self.sub_layers:
            layer.train()
        return self

    def eval(self) -> "ResidualBlock":
        super().eval()
        for layer in self.sub_layers:
            layer.eval()
        return self
