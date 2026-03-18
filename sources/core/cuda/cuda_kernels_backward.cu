#include "core/cuda_kernels.cuh"
#include "core/cuda_kernels_common.cuh"

namespace cvmml {
namespace core {
namespace cuda {

__global__ void add_mul_arrays_kernel(float* grad, const float* grad_y, const float* val, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size ) grad[idx] += grad_y[idx] * val[idx];
}

void add_mul_arrays(float* grad, const float* grad_y, const float* val, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	add_mul_arrays_kernel<<<gridSize, blockSize>>>(grad, grad_y, val, size);
	CHECK_CUDA_LAUNCH();
}

__global__ void add_div_arrays_kernel(float* grad, const float* grad_y, const float* val, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size ) grad[idx] += grad_y[idx] / val[idx];
}

void add_div_arrays(float* grad, const float* grad_y, const float* val, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	add_div_arrays_kernel<<<gridSize, blockSize>>>(grad, grad_y, val, size);
	CHECK_CUDA_LAUNCH();
}

__global__ void sub_mul_div_sqr_arrays_kernel(float* grad_b, const float* grad_y, const float* val_a, const float* val_b, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size ) grad_b[idx] -= (grad_y[idx] * val_a[idx]) / (val_b[idx] * val_b[idx]);
}

void sub_mul_div_sqr_arrays(float* grad_b, const float* grad_y, const float* val_a, const float* val_b, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	sub_mul_div_sqr_arrays_kernel<<<gridSize, blockSize>>>(grad_b, grad_y, val_a, val_b, size);
	CHECK_CUDA_LAUNCH();
}

__global__ void add_mul_scalar_arrays_kernel(float* grad, const float* grad_y, float val, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size ) grad[idx] += grad_y[idx] * val;
}

void add_mul_scalar_arrays(float* grad, const float* grad_y, float val, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	add_mul_scalar_arrays_kernel<<<gridSize, blockSize>>>(grad, grad_y, val, size);
	CHECK_CUDA_LAUNCH();
}

__global__ void add_div_scalar_arrays_kernel(float* grad, const float* grad_y, float val, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size ) grad[idx] += grad_y[idx] / val;
}

void add_div_scalar_arrays(float* grad, const float* grad_y, float val, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	add_div_scalar_arrays_kernel<<<gridSize, blockSize>>>(grad, grad_y, val, size);
	CHECK_CUDA_LAUNCH();
}

} // namespace cuda
} // namespace core
} // namespace cvmml
