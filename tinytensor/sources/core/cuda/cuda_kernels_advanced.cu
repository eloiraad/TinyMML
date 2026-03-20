#include "cuda_kernels.cuh"
#include "cuda_kernels_common.cuh"

namespace tinytensor {
namespace core {
namespace cuda {

__global__ void add_transpose_nd_kernel(const float* grad_y, float* grad_x, const int* parent_shape, int ndim, int dim0, int dim1, int total_size)
{
	int linear_parent = blockIdx.x * blockDim.x + threadIdx.x;
	if ( linear_parent >= total_size ) return;

	int coords[8];
	int rem = linear_parent;
	for ( int d = ndim - 1; d >= 0; --d )
	{
		coords[d] = rem % parent_shape[d];
		rem /= parent_shape[d];
	}

	int tmp = coords[dim0];
	coords[dim0] = coords[dim1];
	coords[dim1] = tmp;

	int y_shape[8];
	for ( int d = 0; d < ndim; ++d ) y_shape[d] = parent_shape[d];
	tmp = y_shape[dim0];
	y_shape[dim0] = y_shape[dim1];
	y_shape[dim1] = tmp;

	int linear_y = 0;
	for ( int d = 0; d < ndim; ++d ) linear_y = linear_y * y_shape[d] + coords[d];
	grad_x[linear_parent] += grad_y[linear_y];
}

void add_transpose_nd(const float* grad_y, float* grad_x, const int* parent_shape, int ndim, int dim0, int dim1, int total_size)
{
	if ( ndim > 8 )
		throw std::invalid_argument("add_transpose_nd currently supports up to 8 dimensions.");

	int* d_shape = nullptr;
	CHECK_CUDA(cudaMalloc(&d_shape, ndim * sizeof(int)));
	CHECK_CUDA(cudaMemcpy(d_shape, parent_shape, ndim * sizeof(int), cudaMemcpyHostToDevice));

	dim3 blockSize, gridSize;
	get_grid_1d(total_size, blockSize, gridSize);
	add_transpose_nd_kernel<<<gridSize, blockSize>>>(grad_y, grad_x, d_shape, ndim, dim0, dim1, total_size);
	CHECK_CUDA_LAUNCH();

	cudaFree(d_shape);
}

__global__ void pack_strided_to_contiguous_kernel(const float* src, float* dst, const int* shape, const int* strides, int ndim, int offset, int total_size)
{
	int linear = blockIdx.x * blockDim.x + threadIdx.x;
	if ( linear >= total_size ) return;

	int rem = linear;
	int source_index = offset;
	for ( int d = ndim - 1; d >= 0; --d )
	{
		int coord = rem % shape[d];
		rem /= shape[d];
		source_index += coord * strides[d];
	}
	dst[linear] = src[source_index];
}

void pack_strided_to_contiguous(const float* src, float* dst, const int* shape, const int* strides, int ndim, int offset, int total_size)
{
	int* d_shape = nullptr;
	int* d_strides = nullptr;
	CHECK_CUDA(cudaMalloc(&d_shape, ndim * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_strides, ndim * sizeof(int)));
	CHECK_CUDA(cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_strides, strides, ndim * sizeof(int), cudaMemcpyHostToDevice));

	dim3 blockSize, gridSize;
	get_grid_1d(total_size, blockSize, gridSize);
	pack_strided_to_contiguous_kernel<<<gridSize, blockSize>>>(src, dst, d_shape, d_strides, ndim, offset, total_size);
	CHECK_CUDA_LAUNCH();

	cudaFree(d_shape);
	cudaFree(d_strides);
}

__global__ void matmult_backward_A_kernel(const float* grad_C, const float* B, float* grad_A, int M, int K, int N)
{
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if ( m < M && k < K )
	{
		float sum = 0.0f;
		for ( int n = 0; n < N; ++n ) sum += grad_C[m * N + n] * B[k * N + n];
		grad_A[m * K + k] += sum;
	}
}

void matmult_backward_A(const float* grad_C, const float* B, float* grad_A, int M, int K, int N)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
	matmult_backward_A_kernel<<<gridSize, blockSize>>>(grad_C, B, grad_A, M, K, N);
	CHECK_CUDA_LAUNCH();
}

__global__ void matmult_backward_B_kernel(const float* A, const float* grad_C, float* grad_B, int M, int K, int N)
{
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if ( k < K && n < N )
	{
		float sum = 0.0f;
		for ( int m = 0; m < M; ++m ) sum += A[m * K + k] * grad_C[m * N + n];
		grad_B[k * N + n] += sum;
	}
}

void matmult_backward_B(const float* A, const float* grad_C, float* grad_B, int M, int K, int N)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (K + blockSize.y - 1) / blockSize.y);
	matmult_backward_B_kernel<<<gridSize, blockSize>>>(A, grad_C, grad_B, M, K, N);
	CHECK_CUDA_LAUNCH();
}

} // namespace cuda
} // namespace core
} // namespace tinytensor
