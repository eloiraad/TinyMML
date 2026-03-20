#include "cuda_kernels.cuh"
#include "cuda_kernels_common.cuh"

#include <climits>
#include <cstdint>
#include <limits>
#include <math.h>

namespace tinytensor {
namespace core {
namespace cuda {

__device__ bool is_reduced_axis(int axis, const int* axes, int num_axes)
{
	for ( int i = 0; i < num_axes; ++i )
		if ( axes[i] == axis ) return true;
	return false;
}

__device__ int map_to_reduced_index(int linear, const int* in_shape, int rank, const int* axes, int num_axes)
{
	int coords[16];
	int rem = linear;
	for ( int d = rank - 1; d >= 0; --d )
	{
		coords[d] = rem % in_shape[d];
		rem /= in_shape[d];
	}

	int out_linear = 0;
	for ( int d = 0; d < rank; ++d )
	{
		if ( is_reduced_axis(d, axes, num_axes) )
			continue;
		out_linear = out_linear * in_shape[d] + coords[d];
	}
	return out_linear;
}

__global__ void reduce_sum_axes_kernel(const float* in, float* out, const int* in_shape, int rank, const int* axes, int num_axes, int in_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= in_size )
		return;
	int out_idx = map_to_reduced_index(i, in_shape, rank, axes, num_axes);
	atomicAdd(&out[out_idx], in[i]);
}

__global__ void reduce_sum_backward_axes_kernel(const float* grad_out, float* grad_in, const int* in_shape, int rank, const int* axes, int num_axes, int in_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= in_size )
		return;
	int out_idx = map_to_reduced_index(i, in_shape, rank, axes, num_axes);
	grad_in[i] += grad_out[out_idx];
}

__device__ uint32_t ordered_bits_from_float(float v)
{
	uint32_t bits = __float_as_uint(v);
	return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ float float_from_ordered_bits(uint32_t ordered)
{
	uint32_t bits = (ordered & 0x80000000u) ? (ordered & 0x7fffffffu) : ~ordered;
	return __uint_as_float(bits);
}

__device__ unsigned long long make_max_key(float value, int index)
{
	uint32_t hi = ordered_bits_from_float(value);
	uint32_t lo = 0xffffffffu - static_cast<uint32_t>(index);
	return (static_cast<unsigned long long>(hi) << 32) | lo;
}

__device__ unsigned long long make_min_key(float value, int index)
{
	uint32_t hi = ordered_bits_from_float(value);
	uint32_t lo = static_cast<uint32_t>(index);
	return (static_cast<unsigned long long>(hi) << 32) | lo;
}

__global__ void init_keys_max_kernel(unsigned long long* keys, int out_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < out_size ) keys[i] = make_max_key(-INFINITY, 0x7fffffff);
}

__global__ void init_keys_min_kernel(unsigned long long* keys, int out_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < out_size ) keys[i] = ULLONG_MAX;
}

__global__ void reduce_max_axes_kernel(const float* in, unsigned long long* out_keys, const int* in_shape, int rank, const int* axes, int num_axes, int in_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= in_size ) return;
	int out_idx = map_to_reduced_index(i, in_shape, rank, axes, num_axes);
	unsigned long long key = make_max_key(in[i], i);
	atomicMax(&out_keys[out_idx], key);
}

__global__ void reduce_min_axes_kernel(const float* in, unsigned long long* out_keys, const int* in_shape, int rank, const int* axes, int num_axes, int in_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= in_size ) return;
	int out_idx = map_to_reduced_index(i, in_shape, rank, axes, num_axes);
	unsigned long long key = make_min_key(in[i], i);
	atomicMin(&out_keys[out_idx], key);
}

__global__ void decode_max_keys_kernel(const unsigned long long* keys, float* out, int* arg, int out_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= out_size ) return;
	unsigned long long key = keys[i];
	uint32_t ordered = static_cast<uint32_t>(key >> 32);
	uint32_t inv_idx = static_cast<uint32_t>(key & 0xffffffffu);
	out[i] = float_from_ordered_bits(ordered);
	arg[i] = static_cast<int>(0xffffffffu - inv_idx);
}

__global__ void decode_min_keys_kernel(const unsigned long long* keys, float* out, int* arg, int out_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= out_size ) return;
	unsigned long long key = keys[i];
	uint32_t ordered = static_cast<uint32_t>(key >> 32);
	uint32_t idx = static_cast<uint32_t>(key & 0xffffffffu);
	out[i] = float_from_ordered_bits(ordered);
	arg[i] = static_cast<int>(idx);
}

__global__ void scatter_backward_kernel(const float* grad_out, float* grad_in, const int* arg, int out_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= out_size ) return;
	int src_idx = arg[i];
	if ( src_idx >= 0 ) atomicAdd(&grad_in[src_idx], grad_out[i]);
}

void reduce_sum_axes(const float* in, float* out, const int* in_shape, int rank, const int* axes, int num_axes, int out_size, int in_size)
{
	if ( rank > 16 )
		throw std::invalid_argument("CUDA reductions currently support up to 16 dimensions.");

	int* d_shape = nullptr;
	int* d_axes = nullptr;
	CHECK_CUDA(cudaMalloc(&d_shape, rank * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_axes, num_axes * sizeof(int)));
	CHECK_CUDA(cudaMemcpy(d_shape, in_shape, rank * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_axes, axes, num_axes * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemset(out, 0, out_size * sizeof(float)));

	dim3 blockSize, gridSize;
	get_grid_1d(in_size, blockSize, gridSize);
	reduce_sum_axes_kernel<<<gridSize, blockSize>>>(in, out, d_shape, rank, d_axes, num_axes, in_size);
	CHECK_CUDA_LAUNCH();

	cudaFree(d_shape);
	cudaFree(d_axes);
}

void reduce_sum_backward_axes(const float* grad_out, float* grad_in, const int* in_shape, int rank, const int* axes, int num_axes, int in_size)
{
	if ( rank > 16 )
		throw std::invalid_argument("CUDA reductions currently support up to 16 dimensions.");

	int* d_shape = nullptr;
	int* d_axes = nullptr;
	CHECK_CUDA(cudaMalloc(&d_shape, rank * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_axes, num_axes * sizeof(int)));
	CHECK_CUDA(cudaMemcpy(d_shape, in_shape, rank * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_axes, axes, num_axes * sizeof(int), cudaMemcpyHostToDevice));

	dim3 blockSize, gridSize;
	get_grid_1d(in_size, blockSize, gridSize);
	reduce_sum_backward_axes_kernel<<<gridSize, blockSize>>>(grad_out, grad_in, d_shape, rank, d_axes, num_axes, in_size);
	CHECK_CUDA_LAUNCH();

	cudaFree(d_shape);
	cudaFree(d_axes);
}

void reduce_max_axes(const float* in, float* out, int* arg_out, const int* in_shape, int rank, const int* axes, int num_axes, int out_size, int in_size)
{
	if ( rank > 16 )
		throw std::invalid_argument("CUDA reductions currently support up to 16 dimensions.");

	int* d_shape = nullptr;
	int* d_axes = nullptr;
	unsigned long long* d_keys = nullptr;
	CHECK_CUDA(cudaMalloc(&d_shape, rank * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_axes, num_axes * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_keys, out_size * sizeof(unsigned long long)));
	CHECK_CUDA(cudaMemcpy(d_shape, in_shape, rank * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_axes, axes, num_axes * sizeof(int), cudaMemcpyHostToDevice));

	dim3 blockOut, gridOut;
	get_grid_1d(out_size, blockOut, gridOut);
	init_keys_max_kernel<<<gridOut, blockOut>>>(d_keys, out_size);

	dim3 blockIn, gridIn;
	get_grid_1d(in_size, blockIn, gridIn);
	reduce_max_axes_kernel<<<gridIn, blockIn>>>(in, d_keys, d_shape, rank, d_axes, num_axes, in_size);
	decode_max_keys_kernel<<<gridOut, blockOut>>>(d_keys, out, arg_out, out_size);
	CHECK_CUDA_LAUNCH();

	cudaFree(d_shape);
	cudaFree(d_axes);
	cudaFree(d_keys);
}

void reduce_min_axes(const float* in, float* out, int* arg_out, const int* in_shape, int rank, const int* axes, int num_axes, int out_size, int in_size)
{
	if ( rank > 16 )
		throw std::invalid_argument("CUDA reductions currently support up to 16 dimensions.");

	int* d_shape = nullptr;
	int* d_axes = nullptr;
	unsigned long long* d_keys = nullptr;
	CHECK_CUDA(cudaMalloc(&d_shape, rank * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_axes, num_axes * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_keys, out_size * sizeof(unsigned long long)));
	CHECK_CUDA(cudaMemcpy(d_shape, in_shape, rank * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_axes, axes, num_axes * sizeof(int), cudaMemcpyHostToDevice));

	dim3 blockOut, gridOut;
	get_grid_1d(out_size, blockOut, gridOut);
	init_keys_min_kernel<<<gridOut, blockOut>>>(d_keys, out_size);

	dim3 blockIn, gridIn;
	get_grid_1d(in_size, blockIn, gridIn);
	reduce_min_axes_kernel<<<gridIn, blockIn>>>(in, d_keys, d_shape, rank, d_axes, num_axes, in_size);
	decode_min_keys_kernel<<<gridOut, blockOut>>>(d_keys, out, arg_out, out_size);
	CHECK_CUDA_LAUNCH();

	cudaFree(d_shape);
	cudaFree(d_axes);
	cudaFree(d_keys);
}

void reduce_scatter_backward(const float* grad_out, float* grad_in, const int* arg_out, int out_size)
{
	dim3 blockSize, gridSize;
	get_grid_1d(out_size, blockSize, gridSize);
	scatter_backward_kernel<<<gridSize, blockSize>>>(grad_out, grad_in, arg_out, out_size);
	CHECK_CUDA_LAUNCH();
}

} // namespace cuda
} // namespace core
} // namespace tinytensor
