"""
Test suite for code review fixes.
Covers: CUDA fast-path for */÷, Linear init, backward() scalar guard,
        subtract_ in-place, SGD optimizer, and CrossEntropyLoss.
"""
import sys
import math
import numpy as np
import cvmml_api as C

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}")


def assert_allclose(name, got, expected, atol=1e-5, rtol=1e-5):
    ok = np.allclose(got, expected, atol=atol, rtol=rtol)
    check(name, ok)
    if not ok:
        print(f"    got:      {got}")
        print(f"    expected: {expected}")


# =========================================================
# 1. CUDA fast-path for operator* and operator/
# =========================================================
print("\n=== CUDA operator* / operator/ fast-path ===")

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

xa = C.Tensor([2, 2]); np.asarray(xa)[:] = a
xb = C.Tensor([2, 2]); np.asarray(xb)[:] = b

# CPU mul/div (baseline)
assert_allclose("cpu mul forward", np.asarray(xa * xb), a * b)
assert_allclose("cpu div forward", np.asarray(xa / xb), a / b)

# CUDA mul/div — these now use cuda::mul_arrays/div_arrays
xac = xa.to_cuda()
xbc = xb.to_cuda()
assert_allclose("cuda mul forward", np.asarray((xac * xbc).to_cpu()), a * b)
assert_allclose("cuda div forward", np.asarray((xac / xbc).to_cpu()), a / b)

# CUDA broadcast still works (falls back to CPU path)
xb_bc = C.Tensor([1, 2]); np.asarray(xb_bc)[:] = np.array([[5.0, 6.0]], dtype=np.float32)
xbc_bc = xb_bc.to_cuda()
assert_allclose("cuda broadcast mul", np.asarray((xac * xbc_bc).to_cpu()),
                a * np.array([[5.0, 6.0]], dtype=np.float32))


# =========================================================
# 2. Linear init: He vs Xavier/default
# =========================================================
print("\n=== Linear init differentiation ===")
sys.path.insert(0, ".")
from nn.model import Linear

lin_he = Linear(16, init="he")
lin_he.build([4, 8])

lin_def = Linear(16, init="default")
lin_def.build([4, 8])

w_he = np.asarray(lin_he.weight)
w_def = np.asarray(lin_def.weight)

# He scale = sqrt(2/8) ≈ 0.5, default scale = sqrt(1/8) ≈ 0.354
he_std = np.std(w_he)
def_std = np.std(w_def)
check("he init has larger scale than default", he_std > def_std * 1.1)
print(f"    he_std={he_std:.4f}, def_std={def_std:.4f}")


# =========================================================
# 3. backward() scalar guard
# =========================================================
print("\n=== backward() scalar guard ===")

x_scalar = C.Tensor([1])
np.asarray(x_scalar)[:] = np.array([3.0], dtype=np.float32)
x_scalar.set_requires_grad(True)
y_scalar = x_scalar * 2.0
y_scalar.backward()
assert_allclose("scalar backward works", np.asarray(x_scalar.grad()), np.array([2.0]))

x_multi = C.Tensor([2, 3])
np.asarray(x_multi)[:] = np.arange(6, dtype=np.float32).reshape(2, 3)
x_multi.set_requires_grad(True)
y_multi = x_multi * 2.0
try:
    y_multi.backward()
    check("non-scalar backward raises error", False)
except RuntimeError as e:
    check("non-scalar backward raises error", "scalar" in str(e).lower())

# Correct pattern: reduce to scalar first
x_r = C.Tensor([2, 3])
np.asarray(x_r)[:] = np.arange(6, dtype=np.float32).reshape(2, 3)
x_r.set_requires_grad(True)
y_r = (x_r * 2.0).sum()
y_r.backward()
assert_allclose("sum then backward works", np.asarray(x_r.grad()), np.full((2, 3), 2.0))


# =========================================================
# 4. subtract_ in-place operation
# =========================================================
print("\n=== subtract_ in-place ===")

xa2 = C.Tensor([2, 2]); np.asarray(xa2)[:] = a.copy()
xb2 = C.Tensor([2, 2]); np.asarray(xb2)[:] = b.copy()
xa2.subtract_(xb2)
assert_allclose("cpu subtract_ in-place", np.asarray(xa2), a - b)

xac2 = C.Tensor([2, 2]); np.asarray(xac2)[:] = a.copy()
xbc2 = C.Tensor([2, 2]); np.asarray(xbc2)[:] = b.copy()
xac2 = xac2.to_cuda()
xbc2 = xbc2.to_cuda()
xac2.subtract_(xbc2)
assert_allclose("cuda subtract_ in-place", np.asarray(xac2.to_cpu()), a - b)


# =========================================================
# 5. SGD optimizer
# =========================================================
print("\n=== SGD optimizer ===")
from nn.sgd import SGD

# Simple test: param starts at [4, 6], grad is [2, 3], lr=0.1
# After step: [4 - 0.1*2, 6 - 0.1*3] = [3.8, 5.7]
param = C.Tensor([2])
np.asarray(param)[:] = np.array([4.0, 6.0], dtype=np.float32)
param.set_requires_grad(True)

# Manually set gradient
y = param.sum()
y.backward()
# grad should be [1, 1]
assert_allclose("sgd pre-step grad", np.asarray(param.grad()), np.array([1.0, 1.0]))

opt = SGD([param], lr=0.1)
opt.step()
assert_allclose("sgd step update", np.asarray(param), np.array([3.9, 5.9]))

opt.zero_grad()
assert_allclose("sgd zero_grad", np.asarray(param.grad()), np.array([0.0, 0.0]))


# =========================================================
# 6. CrossEntropyLoss
# =========================================================
print("\n=== CrossEntropyLoss ===")
from nn.loss import CrossEntropyLoss

# 2 samples, 3 classes
logits_np = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float32)
targets_np = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

# NumPy reference
def numpy_cross_entropy(logits, targets):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return -np.mean(np.sum(targets * log_softmax, axis=1))

expected_loss = numpy_cross_entropy(logits_np, targets_np)

logits_t = C.Tensor([2, 3]); np.asarray(logits_t)[:] = logits_np
targets_t = C.Tensor([2, 3]); np.asarray(targets_t)[:] = targets_np
logits_t.set_requires_grad(True)

ce = CrossEntropyLoss()
loss = ce(logits_t, targets_t)
loss_val = np.asarray(loss)
assert_allclose("cross entropy forward", loss_val, np.array([expected_loss]), atol=1e-4)

# Backward: check gradient shape and that it's non-zero
loss.backward()
grad = np.asarray(logits_t.grad())
check("cross entropy backward non-zero", np.any(np.abs(grad) > 1e-6))
check("cross entropy backward shape", grad.shape == (2, 3))

# Finite difference check for cross entropy
def ce_fn(logits_np_local):
    l = C.Tensor([2, 3]); np.asarray(l)[:] = logits_np_local
    t = C.Tensor([2, 3]); np.asarray(t)[:] = targets_np
    return np.asarray(ce(l, t))[0]

fd_grad = np.zeros_like(logits_np)
eps = 1e-3
for i in range(logits_np.shape[0]):
    for j in range(logits_np.shape[1]):
        tmp = logits_np.copy()
        tmp[i, j] += eps
        fp = ce_fn(tmp)
        tmp[i, j] -= 2 * eps
        fm = ce_fn(tmp)
        fd_grad[i, j] = (fp - fm) / (2 * eps)

assert_allclose("cross entropy grad vs finite diff", grad, fd_grad, atol=1e-3, rtol=1e-3)


# =========================================================
# Summary
# =========================================================
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
