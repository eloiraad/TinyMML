#include "cvmml/core/cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cvmml {
	namespace core {
		namespace cuda {

			__global__ void add_kernel(const float *a, const float *b, float *out, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size )
					out[idx] = a[idx] + b[idx];
			}

			__global__ void sub_kernel(const float *a, const float *b, float *out, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size )
					out[idx] = a[idx] - b[idx];
			}

			__global__ void mul_kernel(const float *a, const float *b, float *out, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size )
					out[idx] = a[idx] * b[idx];
			}

			__global__ void div_kernel(const float *a, const float *b, float *out, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size )
					out[idx] = a[idx] / b[idx];
			}

			__global__ void add_scalar_kernel(const float *a, float scalar, float *out, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size )
					out[idx] = a[idx] + scalar;
			}

			__global__ void sub_scalar_kernel(const float *a, float scalar, float *out, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size )
					out[idx] = a[idx] - scalar;
			}

			__global__ void mul_scalar_kernel(const float *a, float scalar, float *out, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size )
					out[idx] = a[idx] * scalar;
			}

			__global__ void div_scalar_kernel(const float *a, float scalar, float *out, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size )
					out[idx] = a[idx] / scalar;
			}

			__global__ void matmul_kernel(const float *A, const float *B, float *OUT, int M, int K, int N)
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

			inline void get_grid_1d(int size, dim3& blockSize, dim3& gridSize)
			{
				blockSize = dim3(256);
				gridSize = dim3((size + blockSize.x - 1) / blockSize.x);
			}

			void add_arrays(const float *a, const float *b, float *out, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				add_kernel<<<gridSize, blockSize>>>(a, b, out, size);
				cudaDeviceSynchronize();
			}

			void sub_arrays(const float *a, const float *b, float *out, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				sub_kernel<<<gridSize, blockSize>>>(a, b, out, size);
				cudaDeviceSynchronize();
			}

			void mul_arrays(const float *a, const float *b, float *out, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				mul_kernel<<<gridSize, blockSize>>>(a, b, out, size);
				cudaDeviceSynchronize();
			}

			void div_arrays(const float *a, const float *b, float *out, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				div_kernel<<<gridSize, blockSize>>>(a, b, out, size);
				cudaDeviceSynchronize();
			}

			void add_scalar(const float *a, float scalar, float *out, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				add_scalar_kernel<<<gridSize, blockSize>>>(a, scalar, out, size);
				cudaDeviceSynchronize();
			}

			void sub_scalar(const float *a, float scalar, float *out, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				sub_scalar_kernel<<<gridSize, blockSize>>>(a, scalar, out, size);
				cudaDeviceSynchronize();
			}

			void mul_scalar(const float *a, float scalar, float *out, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				mul_scalar_kernel<<<gridSize, blockSize>>>(a, scalar, out, size);
				cudaDeviceSynchronize();
			}

			void div_scalar(const float *a, float scalar, float *out, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				div_scalar_kernel<<<gridSize, blockSize>>>(a, scalar, out, size);
				cudaDeviceSynchronize();
			}

			void matmul(const float *a, const float *b, float *out, int M, int K, int N)
			{
				dim3 blockSize(16, 16);
				dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
				matmul_kernel<<<gridSize, blockSize>>>(a, b, out, M, K, N);
				cudaDeviceSynchronize();
			}

			__global__ void fill_ones_kernel(float* ptr, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if ( idx < size ) ptr[idx] = 1.0f;
			}

			float* allocate_memory(int size)
			{
				float* ptr = nullptr;
				if (cudaMalloc(&ptr, size * sizeof(float)) != cudaSuccess)
					throw std::runtime_error("CUDA malloc failed");
				return ptr;
			}

			void free_memory(float* ptr)
			{
				cudaFree(ptr);
			}

			void set_memory(float* ptr, float val, int size)
			{
				if (val == 0.0f)
					cudaMemset(ptr, 0, size * sizeof(float));
			}

			void copy_to_device(float* dst, const float* src, int size)
			{
				cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyHostToDevice);
			}

			void copy_to_host(float* dst, const float* src, int size)
			{
				cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToHost);
			}

			void fill_ones(float* ptr, int size)
			{
				dim3 blockSize, gridSize;
				get_grid_1d(size, blockSize, gridSize);
				fill_ones_kernel<<<gridSize, blockSize>>>(ptr, size);
				cudaDeviceSynchronize();
			}

			__global__ void add_mul_arrays_kernel(float* grad, const float* grad_y, const float* val, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx < size)
					grad[idx] += grad_y[idx] * val[idx];
			}

			void add_mul_arrays(float* grad, const float* grad_y, const float* val, int size)
			{
				dim3 blockSize, gridSize; get_grid_1d(size, blockSize, gridSize);
				add_mul_arrays_kernel<<<gridSize, blockSize>>>(grad, grad_y, val, size);
				cudaDeviceSynchronize();
			}

			__global__ void add_div_arrays_kernel(float* grad, const float* grad_y, const float* val, int size)
			{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx < size)
				grad[idx] += grad_y[idx] / val[idx];
			}

			void add_div_arrays(float* grad, const float* grad_y, const float* val, int size)
			{
				dim3 blockSize, gridSize; get_grid_1d(size, blockSize, gridSize);
				add_div_arrays_kernel<<<gridSize, blockSize>>>(grad, grad_y, val, size);
				cudaDeviceSynchronize();
			}

			__global__ void sub_mul_div_sqr_arrays_kernel(float* grad_b, const float* grad_y, const float* val_a, const float* val_b, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx < size)
					grad_b[idx] -= (grad_y[idx] * val_a[idx]) / (val_b[idx] * val_b[idx]);
			}

			void sub_mul_div_sqr_arrays(float* grad_b, const float* grad_y, const float* val_a, const float* val_b, int size)
			{
				dim3 blockSize, gridSize; get_grid_1d(size, blockSize, gridSize);
				sub_mul_div_sqr_arrays_kernel<<<gridSize, blockSize>>>(grad_b, grad_y, val_a, val_b, size);
				cudaDeviceSynchronize();
			}

			__global__ void add_mul_scalar_arrays_kernel(float* grad, const float* grad_y, float val, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx < size)
					grad[idx] += grad_y[idx] * val;
			}

			void add_mul_scalar_arrays(float* grad, const float* grad_y, float val, int size)
			{
				dim3 blockSize, gridSize; get_grid_1d(size, blockSize, gridSize);
				add_mul_scalar_arrays_kernel<<<gridSize, blockSize>>>(grad, grad_y, val, size);
				cudaDeviceSynchronize();
			}

			__global__ void add_div_scalar_arrays_kernel(float* grad, const float* grad_y, float val, int size)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx < size)
					grad[idx] += grad_y[idx] / val;
			}

			void add_div_scalar_arrays(float* grad, const float* grad_y, float val, int size)
			{
				dim3 blockSize, gridSize; get_grid_1d(size, blockSize, gridSize);
				add_div_scalar_arrays_kernel<<<gridSize, blockSize>>>(grad, grad_y, val, size);
				cudaDeviceSynchronize();
			}

			__global__ void transpose_matrix_kernel(const float* src, float* dst, int M, int N)
			{
				int i = blockIdx.y * blockDim.y + threadIdx.y;
				int j = blockIdx.x * blockDim.x + threadIdx.x;
				if (i < M && j < N)
					dst[j * M + i] = src[i * N + j];
			}

			void transpose_matrix(const float* src, float* dst, int M, int N)
			{
				dim3 blockSize(16, 16);
				dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
				transpose_matrix_kernel<<<gridSize, blockSize>>>(src, dst, M, N);
				cudaDeviceSynchronize();
			}

			__global__ void add_transpose_matrix_kernel(const float* grad_y, float* grad_x, int M, int N)
			{
				int i = blockIdx.y * blockDim.y + threadIdx.y;
				int j = blockIdx.x * blockDim.x + threadIdx.x;
				if (i < M && j < N)
					grad_x[i * N + j] += grad_y[j * M + i];
			}

			void add_transpose_matrix(const float* grad_y, float* grad_x, int M, int N)
			{
				dim3 blockSize(16, 16);
				dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
				add_transpose_matrix_kernel<<<gridSize, blockSize>>>(grad_y, grad_x, M, N);
				cudaDeviceSynchronize();
			}

			__global__ void matmult_backward_A_kernel(const float* grad_C, const float* B, float* grad_A, int M, int K, int N)
			{
				int m = blockIdx.y * blockDim.y + threadIdx.y;
				int k = blockIdx.x * blockDim.x + threadIdx.x;
				if (m < M && k < K)
				{
					float sum = 0.0f;
					for (int n = 0; n < N; ++n)
						sum += grad_C[m * N + n] * B[k * N + n]; // dA = dC * B^T
					grad_A[m * K + k] += sum;
				}
			}

			void matmult_backward_A(const float* grad_C, const float* B, float* grad_A, int M, int K, int N)
			{
				dim3 blockSize(16, 16);
				dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
				matmult_backward_A_kernel<<<gridSize, blockSize>>>(grad_C, B, grad_A, M, K, N);
				cudaDeviceSynchronize();
			}

			__global__ void matmult_backward_B_kernel(const float* A, const float* grad_C, float* grad_B, int M, int K, int N)
			{
				int k = blockIdx.y * blockDim.y + threadIdx.y;
				int n = blockIdx.x * blockDim.x + threadIdx.x;
				if (k < K && n < N)
				{
					float sum = 0.0f;
					for (int m = 0; m < M; ++m)
						sum += A[m * K + k] * grad_C[m * N + n]; // dB = A^T * dC
					grad_B[k * N + n] += sum;
				}
			}

			void matmult_backward_B(const float* A, const float* grad_C, float* grad_B, int M, int K, int N)
			{
				dim3 blockSize(16, 16);
				dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (K + blockSize.y - 1) / blockSize.y);
				matmult_backward_B_kernel<<<gridSize, blockSize>>>(A, grad_C, grad_B, M, K, N);
				cudaDeviceSynchronize();
			}

		} // namespace cuda
	} // namespace core
} // namespace cvmml
