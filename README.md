# CV-ML

## Core Architecture (Post-Refactor)

- `sources/core/tensor/`
	- `tensor_memory.cpp`: allocation/layout/views/contiguous/indexing
	- `tensor_broadcast.cpp`: shared shape/index/axes utilities
	- `tensor_ops_elementwise.cpp`: `+ - * /`, `exp`, `log` (+ backward)
	- `tensor_ops_reduction.cpp`: `sum`, `max`, `min` (+ backward)
	- `tensor_ops_linalg.cpp`: `matmult`, `transpose` (+ backward)
	- `tensor_autograd.cpp`: graph flags, buffers, topo `backward`
- `sources/core/cuda/`
	- `cuda_kernels_elementwise.cu`: elementwise/scalar/unary kernels + matmul forward
	- `cuda_kernels_backward.cu`: reusable backward helpers
	- `cuda_kernels_reduction.cu`: `sum/max/min` reductions + backward scatter
	- `cuda_kernels_advanced.cu`: transpose backward, strided pack, matmul backward

## Brief Convention

For non-trivial functions, use a short in-file brief:
- `What`: what the function does.
- `Why`: why this behavior/constraint exists.

Keep briefs concise (2 lines) and colocated with the implementation.

## Product Decisions

- This refactor is delivered in one global wave (big-bang).
- CPU multithreading is intentionally deferred to a later profiling-gated phase.