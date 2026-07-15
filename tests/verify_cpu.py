"""CPU verification for TinyMML.

Run from repository root with ``.venv/bin/python tests/verify_cpu.py``.
Each section prints its purpose and raises immediately with useful context.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tinytensor as tt
import tinymodel as tm


def section(name: str, check: Callable[[], None]) -> None:
    """Run one diagnostic section and print an explicit result."""
    print(f"\n=== {name} ===")
    try:
        check()
    except Exception as error:
        print(f"FAIL: {error}")
        raise
    print("PASS")


def tensor_from(values: np.ndarray) -> tt.Tensor:
    """Create a CPU tensor containing contiguous float32 values."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    tensor = tt.Tensor(list(values.shape))
    np.asarray(tensor)[:] = values
    return tensor


def check_environment() -> None:
    """Report interpreter, platform, and compiled backend support."""
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA compiled: {tt.cuda_available()}")
    assert hasattr(tt, "Tensor"), "tinytensor extension did not expose Tensor"
    assert tt.cuda_available() is False, "CPU verification requires CPU-only build"


def check_tensor_and_numpy() -> None:
    """Verify buffer interop, broadcasting, reduction, and matrix multiply."""
    left_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    right_np = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    left = tensor_from(left_np)
    right = tensor_from(right_np)

    np.testing.assert_allclose(np.asarray(left + right), left_np + right_np)
    np.testing.assert_allclose(np.asarray(left.sum([1], True)), left_np.sum(1, keepdims=True))

    matrix_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    matrix = tensor_from(matrix_np)
    np.testing.assert_allclose(np.asarray(left.mm(matrix)), left_np @ matrix_np)


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Estimate one scalar function gradient with centered differences."""
    gradient = np.zeros_like(values, dtype=np.float32)
    for index in np.ndindex(values.shape):
        positive = values.copy()
        negative = values.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        gradient[index] = (function(positive) - function(negative)) / (2 * epsilon)
    return gradient


def check_gradient_numerics() -> None:
    """Compare autograd with finite differences for core differentiable ops."""
    values = np.array([[-0.4, 0.2], [0.7, 1.1]], dtype=np.float32)
    tensor = tensor_from(values)
    tensor.set_requires_grad(True)
    loss = (tensor * tensor + tensor).mean([0, 1], False)
    loss.backward()

    def elementwise_objective(candidate: np.ndarray) -> float:
        return float(np.mean(candidate * candidate + candidate))

    np.testing.assert_allclose(
        tensor.grad(),
        numerical_gradient(elementwise_objective, values),
        rtol=2e-3,
        atol=2e-3,
    )

    left_values = np.array([[0.2, -0.3, 0.5], [1.0, 0.4, -0.2]], dtype=np.float32)
    right_values = np.array([[0.7, -0.1], [0.2, 0.8], [-0.6, 0.3]], dtype=np.float32)
    left = tensor_from(left_values)
    left.set_requires_grad(True)
    left.mm(tensor_from(right_values)).sum().backward()

    def matmul_objective(candidate: np.ndarray) -> float:
        return float(np.sum(candidate @ right_values))

    np.testing.assert_allclose(
        left.grad(),
        numerical_gradient(matmul_objective, left_values),
        rtol=2e-3,
        atol=2e-3,
    )


def check_view_autograd() -> None:
    """Finite-difference graph paths through view, transpose, and contiguous."""
    source_values = np.arange(6, dtype=np.float32).reshape(2, 3) / 5.0
    transpose_weights = np.array(
        [[0.3, -0.8], [1.2, 0.5], [-0.4, 0.9]],
        dtype=np.float32,
    )
    source = tensor_from(source_values)
    source.set_requires_grad(True)
    packed = source.T(0, 1).contiguous().view([3, 2])
    (packed * tensor_from(transpose_weights)).sum().backward()

    def transpose_objective(candidate: np.ndarray) -> float:
        return float(np.sum(candidate.T.reshape(3, 2) * transpose_weights))

    np.testing.assert_allclose(
        source.grad(),
        numerical_gradient(transpose_objective, source_values),
        rtol=2e-3,
        atol=2e-3,
    )

    view_weights = np.array(
        [[-0.2, 0.7], [1.1, -0.5], [0.4, 0.9]],
        dtype=np.float32,
    )
    source = tensor_from(source_values)
    source.set_requires_grad(True)
    (source.view([3, 2]) * tensor_from(view_weights)).sum().backward()

    def view_objective(candidate: np.ndarray) -> float:
        return float(np.sum(candidate.reshape(3, 2) * view_weights))

    np.testing.assert_allclose(
        source.grad(),
        numerical_gradient(view_objective, source_values),
        rtol=2e-3,
        atol=2e-3,
    )


def check_im2col_autograd() -> None:
    """Compare im2col backward scatter with centered finite differences."""
    image_values = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 8.0
    column_weights = np.array(
        [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8],
          [0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]],
        dtype=np.float32,
    )
    image = tensor_from(image_values)
    image.set_requires_grad(True)
    columns = image.im2col(2, 2, 1, 0)
    (columns * tensor_from(column_weights)).sum().backward()

    def im2col_objective(candidate: np.ndarray) -> float:
        patches = []
        for kernel_row in range(2):
            for kernel_column in range(2):
                patch = candidate[
                    :, :, kernel_row : kernel_row + 2, kernel_column : kernel_column + 2
                ]
                patches.append(patch.reshape(1, -1))
        lowered = np.stack(patches, axis=1)
        return float(np.sum(lowered * column_weights))

    np.testing.assert_allclose(
        image.grad(),
        numerical_gradient(im2col_objective, image_values),
        rtol=2e-3,
        atol=2e-3,
    )


def check_layers_and_conv2d() -> None:
    """Verify layer shapes and a complete Conv2D gradient path."""
    dense_input = tensor_from(np.arange(6, dtype=np.float32).reshape(2, 3))
    dense_output = tm.ReLU()(tm.Linear(4)(dense_input))
    assert dense_output.shape() == [2, 4]

    reference_input = tensor_from(np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3))
    reference_convolution = tm.Conv2D(1, kernel_size=2, bias=False)
    reference_convolution.build([1, 1, 3, 3])
    np.asarray(reference_convolution.weight)[:] = [[1.0, 0.0, 0.0, -1.0]]
    reference_output = reference_convolution(reference_input)
    expected_output = np.full((1, 1, 2, 2), -4.0, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(reference_output), expected_output)

    images = tensor_from(np.arange(50, dtype=np.float32).reshape(2, 1, 5, 5) / 50.0)
    images.set_requires_grad(True)

    convolution = tm.Conv2D(2, kernel_size=3, stride=2, padding=1)
    flattened = tm.Flatten()
    features = convolution(images).relu()
    output = flattened(features)

    assert output.shape() == [2, 18]
    output.sum().backward()
    assert convolution.weight.grad() is not None
    assert np.any(np.abs(convolution.weight.grad()) > 0.0)
    assert np.any(np.abs(images.grad()) > 0.0)

    before = np.array(convolution.weight, copy=True)
    optimizer = tm.SGD(convolution.parameters(), lr=0.01)
    optimizer.step()
    assert not np.allclose(before, np.asarray(convolution.weight))


def check_training_pipeline() -> None:
    """Fit deterministic classification and regression examples."""
    tt.manual_seed(42)
    np.random.seed(42)

    x_class = np.array(
        [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]],
        dtype=np.float32,
    )
    y_class = np.eye(4, dtype=np.float32)
    classifier = tm.Model([tm.Linear(8), tm.ReLU(), tm.Linear(4)])
    classifier.build(list(x_class.shape))
    history = classifier.fit(
        x_class,
        y_class,
        epochs=40,
        batch_size=4,
        optimizer=tm.Adam(classifier.parameters(), lr=0.1),
        loss_fn=tm.CrossEntropyLoss(),
        shuffle=False,
        metric="accuracy",
        verbose=False,
    )
    assert history["metric"][-1] == 1.0
    assert history["loss"][-1] < history["loss"][0]

    x_reg = np.linspace(-1, 1, 20, dtype=np.float32).reshape(-1, 1)
    y_reg = 2.0 * x_reg + 1.0
    regressor = tm.Model([tm.Linear(1)])
    regressor.build(list(x_reg.shape))
    history = regressor.fit(
        x_reg,
        y_reg,
        epochs=50,
        batch_size=10,
        optimizer=tm.Adam(regressor.parameters(), lr=0.1),
        loss_fn=tm.MSELoss(),
        shuffle=False,
        verbose=False,
    )
    assert history["loss"][-1] < 0.02


def check_validation_errors() -> None:
    """Verify invalid public inputs fail with actionable messages."""
    try:
        tm.Model(device="metal")
    except ValueError as error:
        assert "cpu" in str(error).lower() and "cuda" in str(error).lower()
    else:
        raise AssertionError("Model accepted unsupported device 'metal'")

    tensor = tt.Tensor([1])
    try:
        tensor.to_cuda()
    except RuntimeError as error:
        assert "--cuda" in str(error)
    else:
        raise AssertionError("CPU-only build allowed to_cuda()")

    model = tm.Model([tm.Linear(1)])
    model.build([2, 1])
    optimizer = tm.SGD(model.parameters(), lr=0.01)
    loss = tm.MSELoss()
    invalid_fit_cases = [
        (np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32), 1, 1, "sample"),
        (np.ones((2, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32), 1, 1, "same"),
        (np.ones((2, 1), dtype=np.float32), np.ones((2, 1), dtype=np.float32), 0, 1, "epoch"),
        (np.ones((2, 1), dtype=np.float32), np.ones((2, 1), dtype=np.float32), 1, 0, "batch"),
    ]
    for x_data, y_data, epochs, batch_size, expected_message in invalid_fit_cases:
        try:
            model.fit(
                x_data,
                y_data,
                epochs=epochs,
                batch_size=batch_size,
                optimizer=optimizer,
                loss_fn=loss,
                verbose=False,
            )
        except ValueError as error:
            assert expected_message in str(error).lower()
        else:
            raise AssertionError(f"fit accepted invalid input expected to mention {expected_message}")


def main() -> None:
    """Run core verification sections in dependency order."""
    section("Environment and backend", check_environment)
    section("Tensor and NumPy behavior", check_tensor_and_numpy)
    section("Finite-difference gradients", check_gradient_numerics)
    section("View autograd", check_view_autograd)
    section("im2col autograd", check_im2col_autograd)
    section("Layers and Conv2D autograd", check_layers_and_conv2d)
    section("Training pipelines", check_training_pipeline)
    section("Validation errors", check_validation_errors)
    print("\nCORE CPU VERIFICATION PASSED")


if __name__ == "__main__":
    main()
