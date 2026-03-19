#include "core/cuda_kernels.cuh"
#include "core/cuda_kernels_common.cuh"

namespace cvmml {
namespace core {
namespace cuda {

__global__ void add_kernel(const float* a, const float* b, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = a[idx] + b[idx];
}

__global__ void sub_kernel(const float* a, const float* b, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = a[idx] - b[idx];
}

__global__ void mul_kernel(const float* a, const float* b, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = a[idx] * b[idx];
}

__global__ void div_kernel(const float* a, const float* b, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = a[idx] / b[idx];
}

__global__ void add_scalar_kernel(const float* a, float scalar, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = a[idx] + scalar;
}

__global__ void sub_scalar_kernel(const float* a, float scalar, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = a[idx] - scalar;
}

__global__ void mul_scalar_kernel(const float* a, float scalar, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = a[idx] * scalar;
}

__global__ void div_scalar_kernel(const float* a, float scalar, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = a[idx] / scalar;
}

__global__ void exp_kernel(const float* a, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = expf(a[idx]);
}

__global__ void log_kernel(const float* a, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = logf(a[idx]);
}

__global__ void matmul_kernel(const float* A, const float* B, float* OUT, int M, int K, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < M && col < N )
	{
		float sum = 0.0f;
		for ( int k = 0; k < K; ++k )
			sum += A[row * K + k] * B[k * N + col];
		OUT[row * N + col] = sum;
	}
}

__global__ void fill_ones_kernel(float* ptr, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		ptr[idx] = 1.0f;
}

__global__ void relu_kernel(const float* a, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = (a[idx] > 0.0f) ? a[idx] : 0.0f;
}

__global__ void relu_backward_kernel(float* grad, const float* grad_y, const float* x, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		grad[idx] += (x[idx] > 0.0f) ? grad_y[idx] : 0.0f;
}

__global__ void sqrt_kernel(const float* a, float* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
		out[idx] = sqrtf(a[idx]);
}

__global__ void sqrt_backward_kernel(float* grad, const float* grad_y, const float* y_sqrt, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < size )
	{
		float y = y_sqrt[idx];
		if ( y > 0.0f )
			grad[idx] += grad_y[idx] * (0.5f / y);
	}
}

void add_arrays(const float* a, const float* b, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	add_kernel<<<gridSize, blockSize>>>(a, b, out, size);
	CHECK_CUDA_LAUNCH();
}

void sub_arrays(const float* a, const float* b, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	sub_kernel<<<gridSize, blockSize>>>(a, b, out, size);
	CHECK_CUDA_LAUNCH();
}

void mul_arrays(const float* a, const float* b, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	mul_kernel<<<gridSize, blockSize>>>(a, b, out, size);
	CHECK_CUDA_LAUNCH();
}

void div_arrays(const float* a, const float* b, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	div_kernel<<<gridSize, blockSize>>>(a, b, out, size);
	CHECK_CUDA_LAUNCH();
}

void add_scalar(const float* a, float scalar, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	add_scalar_kernel<<<gridSize, blockSize>>>(a, scalar, out, size);
	CHECK_CUDA_LAUNCH();
}

void sub_scalar(const float* a, float scalar, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	sub_scalar_kernel<<<gridSize, blockSize>>>(a, scalar, out, size);
	CHECK_CUDA_LAUNCH();
}

void mul_scalar(const float* a, float scalar, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	mul_scalar_kernel<<<gridSize, blockSize>>>(a, scalar, out, size);
	CHECK_CUDA_LAUNCH();
}

void div_scalar(const float* a, float scalar, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	div_scalar_kernel<<<gridSize, blockSize>>>(a, scalar, out, size);
	CHECK_CUDA_LAUNCH();
}

void exp_array(const float* a, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	exp_kernel<<<gridSize, blockSize>>>(a, out, size);
	CHECK_CUDA_LAUNCH();
}

void log_array(const float* a, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	log_kernel<<<gridSize, blockSize>>>(a, out, size);
	CHECK_CUDA_LAUNCH();
}

void relu_array(const float* a, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	relu_kernel<<<gridSize, blockSize>>>(a, out, size);
	CHECK_CUDA_LAUNCH();
}

void relu_backward_array(float* grad, const float* grad_y, const float* x, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	relu_backward_kernel<<<gridSize, blockSize>>>(grad, grad_y, x, size);
	CHECK_CUDA_LAUNCH();
}

void sqrt_array(const float* a, float* out, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	sqrt_kernel<<<gridSize, blockSize>>>(a, out, size);
	CHECK_CUDA_LAUNCH();
}

void sqrt_backward_array(float* grad, const float* grad_y, const float* y_sqrt, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	sqrt_backward_kernel<<<gridSize, blockSize>>>(grad, grad_y, y_sqrt, size);
	CHECK_CUDA_LAUNCH();
}

void matmul(const float* a, const float* b, float* out, int M, int K, int N)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
	matmul_kernel<<<gridSize, blockSize>>>(a, b, out, M, K, N);
	CHECK_CUDA_LAUNCH();
}

float* allocate_memory(int size)
{
	float* ptr = nullptr;
	CHECK_CUDA(cudaMalloc(&ptr, size * sizeof(float)));
	return ptr;
}

int* allocate_int_memory(int size)
{
	int* ptr = nullptr;
	CHECK_CUDA(cudaMalloc(&ptr, size * sizeof(int)));
	return ptr;
}

void free_memory(float* ptr)
{
	cudaFree(ptr);
}

void free_int_memory(int* ptr)
{
	cudaFree(ptr);
}

void set_memory(float* ptr, float val, int size)
{
	if ( val == 0.0f )
		CHECK_CUDA(cudaMemset(ptr, 0, size * sizeof(float)));
}

void copy_to_device(float* dst, const float* src, int size)
{
	CHECK_CUDA(cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyHostToDevice));
}

void copy_to_host(float* dst, const float* src, int size)
{
	CHECK_CUDA(cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void fill_ones(float* ptr, int size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(size, blockSize, gridSize);
	fill_ones_kernel<<<gridSize, blockSize>>>(ptr, size);
	CHECK_CUDA_LAUNCH();
}

} // namespace cuda
} // namespace core
} // namespace cvmml
