#ifndef CVMML_CORE_CUDA_KERNELS_CUH
#define CVMML_CORE_CUDA_KERNELS_CUH

namespace cvmml {
	namespace core {
		namespace cuda {

			float* allocate_memory(int size);
			void free_memory(float* ptr);
			void set_memory(float* ptr, float val, int size);
			void copy_to_device(float* dst, const float* src, int size);
			void copy_to_host(float* dst, const float* src, int size);
			void fill_ones(float* ptr, int size);

			void add_arrays(const float *a, const float *b, float *out, int size);
			void sub_arrays(const float *a, const float *b, float *out, int size);
			void mul_arrays(const float *a, const float *b, float *out, int size);
			void div_arrays(const float *a, const float *b, float *out, int size);

			void add_scalar(const float *a, float scalar, float *out, int size);
			void sub_scalar(const float *a, float scalar, float *out, int size);
			void mul_scalar(const float *a, float scalar, float *out, int size);
			void div_scalar(const float *a, float scalar, float *out, int size);

			void add_mul_arrays(float* grad, const float* grad_y, const float* val, int size);
			void add_div_arrays(float* grad, const float* grad_y, const float* val, int size);
			void sub_mul_div_sqr_arrays(float* grad_b, const float* grad_y, const float* val_a, const float* val_b, int size);

			void add_mul_scalar_arrays(float* grad, const float* grad_y, float val, int size);
			void add_div_scalar_arrays(float* grad, const float* grad_y, float val, int size);

			void transpose_matrix(const float* src, float* dst, int M, int N);
			void add_transpose_matrix(const float* grad_y, float* grad_x, int M, int N);

			void matmult_backward_A(const float* grad_C, const float* B, float* grad_A, int M, int K, int N);
			void matmult_backward_B(const float* A, const float* grad_C, float* grad_B, int M, int K, int N);
			void pack_strided_to_contiguous(const float* src, float* dst, const int* shape, const int* strides, int ndim, int offset, int total_size);

			void matmul(const float *a, const float *b, float *out, int M, int K, int N);

		} // namespace cuda
	} // namespace core
} // namespace cvmml

#endif // CVMML_CORE_CUDA_KERNELS_CUH