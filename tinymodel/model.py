"""Sequential model container and training loops."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import numpy as np
import tinytensor as tt

from .layer import Layer


Metric = Optional[str | Callable[[np.ndarray, np.ndarray], float]]


class Model(Layer):
    """Keras-style sequential container for TinyMML layers."""

    def __init__(
        self,
        layers: Optional[Iterable[Layer]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        normalized_device = device.lower()
        if normalized_device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if normalized_device == "cuda" and not tt.cuda_available():
            raise RuntimeError(
                "TinyMML was built without CUDA support. Rebuild with './build.sh --cuda'."
            )
        self.layers = list(layers) if layers is not None else []
        self.device = normalized_device

    def add(self, layer: Layer) -> None:
        """Append a layer and invalidate cached shape information."""
        self.layers.append(layer)
        self._built = False

    def _build(self, input_shape):
        current_shape = list(input_shape) if input_shape is not None else None
        for layer in self.layers:
            current_shape = (
                layer.build(current_shape) if not layer.built else layer.output_shape
            )
        return current_shape

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def __call__(self, x):
        if not self.built:
            self.build(x.shape() if hasattr(x, "shape") else None)
        return self.forward(x)

    def parameters(self) -> list:
        """Return trainable tensors from every child layer."""
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters

    def train(self) -> "Model":
        super().train()
        for layer in self.layers:
            layer.train()
        return self

    def eval(self) -> "Model":
        super().eval()
        for layer in self.layers:
            layer.eval()
        return self

    def _np_to_tensor(self, values: np.ndarray) -> tt.Tensor:
        values = np.ascontiguousarray(values, dtype=np.float32)
        tensor = tt.Tensor(list(values.shape))
        np.asarray(tensor)[:] = values
        return tensor.to_cuda() if self.device == "cuda" else tensor

    @staticmethod
    def _tensor_to_np(tensor) -> np.ndarray:
        if tensor.device() == tt.Device.CUDA:
            tensor = tensor.to_cpu()
        return np.array(tensor, dtype=np.float32, copy=True)

    @staticmethod
    def _compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        if predictions.shape[1] > 1:
            predicted_labels = np.argmax(predictions, axis=1)
            target_labels = np.argmax(targets, axis=1)
        else:
            predicted_labels = (predictions[:, 0] >= 0.5).astype(np.int32)
            target_labels = targets[:, 0].astype(np.int32)
        return float(np.mean(predicted_labels == target_labels))

    @classmethod
    def _eval_metric(
        cls,
        predictions: np.ndarray,
        targets: np.ndarray,
        metric: Metric,
    ) -> Optional[float]:
        if metric is None:
            return None
        if metric == "accuracy":
            return cls._compute_accuracy(predictions, targets)
        if callable(metric):
            return float(metric(predictions, targets))
        raise ValueError("metric must be 'accuracy', a callable, or None")

    @staticmethod
    def _validate_dataset(
        x_data: np.ndarray,
        y_data: Optional[np.ndarray] = None,
    ) -> None:
        if x_data.ndim == 0 or x_data.shape[0] == 0:
            raise ValueError("x_data must contain at least one sample")
        if y_data is not None and y_data.shape[0] != x_data.shape[0]:
            raise ValueError("x_data and y_data must contain the same number of samples")

    def fit(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        epochs: int,
        batch_size: int,
        optimizer,
        loss_fn,
        shuffle: bool = True,
        metric: Metric = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train with mini-batch gradient descent and return epoch history."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        x_data = np.asarray(x_data, dtype=np.float32)
        y_data = np.asarray(y_data, dtype=np.float32)
        self._validate_dataset(x_data, y_data)
        sample_count = x_data.shape[0]

        history: dict[str, list[float]] = {"loss": []}
        if metric is not None:
            history["metric"] = []

        self.train()
        for epoch in range(epochs):
            indices = np.arange(sample_count)
            if shuffle:
                np.random.shuffle(indices)

            weighted_loss = 0.0
            epoch_predictions = []
            epoch_targets = []

            for start in range(0, sample_count, batch_size):
                batch_indices = indices[start : start + batch_size]
                x_batch = self._np_to_tensor(x_data[batch_indices])
                y_batch = self._np_to_tensor(y_data[batch_indices])

                optimizer.zero_grad()
                predictions = self(x_batch)
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                optimizer.step()

                current_batch_size = len(batch_indices)
                weighted_loss += self._tensor_to_np(loss).item() * current_batch_size
                if metric is not None:
                    epoch_predictions.append(self._tensor_to_np(predictions))
                    epoch_targets.append(y_data[batch_indices])

            average_loss = weighted_loss / sample_count
            history["loss"].append(average_loss)

            metric_value = None
            if metric is not None:
                metric_value = self._eval_metric(
                    np.concatenate(epoch_predictions),
                    np.concatenate(epoch_targets),
                    metric,
                )
                history["metric"].append(metric_value)

            if verbose:
                message = f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}"
                if metric_value is not None:
                    metric_name = metric if isinstance(metric, str) else "metric"
                    message += f" - {metric_name}: {metric_value:.4f}"
                print(message)

        return history

    def evaluate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        loss_fn,
        batch_size: Optional[int] = None,
        metric: Metric = None,
        return_outputs: bool = False,
    ):
        """Evaluate loss and an optional metric without updating parameters."""
        x_data = np.asarray(x_data, dtype=np.float32)
        y_data = np.asarray(y_data, dtype=np.float32)
        self._validate_dataset(x_data, y_data)
        sample_count = x_data.shape[0]
        batch_size = sample_count if batch_size is None else batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.eval()
        weighted_loss = 0.0
        output_batches = []
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            predictions = self(self._np_to_tensor(x_data[start:end]))
            loss = loss_fn(predictions, self._np_to_tensor(y_data[start:end]))
            weighted_loss += self._tensor_to_np(loss).item() * (end - start)
            output_batches.append(self._tensor_to_np(predictions))

        outputs = np.concatenate(output_batches)
        average_loss = weighted_loss / sample_count
        metric_value = self._eval_metric(outputs, y_data, metric)
        metric_value = 0.0 if metric_value is None else metric_value
        if return_outputs:
            return average_loss, metric_value, outputs
        return average_loss, metric_value

    def predict(
        self,
        x_data: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Return model outputs for all samples."""
        x_data = np.asarray(x_data, dtype=np.float32)
        self._validate_dataset(x_data)
        sample_count = x_data.shape[0]
        batch_size = sample_count if batch_size is None else batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.eval()
        outputs = []
        for start in range(0, sample_count, batch_size):
            predictions = self(self._np_to_tensor(x_data[start : start + batch_size]))
            outputs.append(self._tensor_to_np(predictions))
        return np.concatenate(outputs)

    def get_layer_weights(self) -> dict[str, np.ndarray]:
        """Return copies of layer weights and biases for inspection."""
        weights = {}
        for index, layer in enumerate(self.layers):
            layer_name = type(layer).__name__
            for attribute in ("weight", "bias"):
                parameter = getattr(layer, attribute, None)
                if parameter is None:
                    continue
                if parameter.device() == tt.Device.CUDA:
                    parameter = parameter.to_cpu()
                weights[f"{index}_{layer_name}_{attribute}"] = np.array(
                    parameter,
                    dtype=np.float32,
                    copy=True,
                )
        return weights
