#ifndef CVMML_CORE_CUDA_KERNELS_CUH
#define CVMML_CORE_CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <math.h>

namespace cvmml {
namespace core {
namespace cuda {

float* allocate_memory(int size);
int* allocate_int_memory(int size);
void free_memory(float* ptr);
void free_int_memory(int* ptr);
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
void exp_array(const float *a, float *out, int size);
void log_array(const float *a, float *out, int size);

void add_mul_arrays(float* grad, const float* grad_y, const float* val, int size);
void add_div_arrays(float* grad, const float* grad_y, const float* val, int size);
void sub_mul_div_sqr_arrays(float* grad_b, const float* grad_y, const float* val_a, const float* val_b, int size);

void add_mul_scalar_arrays(float* grad, const float* grad_y, float val, int size);
void add_div_scalar_arrays(float* grad, const float* grad_y, float val, int size);

void add_transpose_nd(const float* grad_y, float* grad_x, const int* parent_shape, int ndim, int dim0, int dim1, int total_size);

void matmult_backward_A(const float* grad_C, const float* B, float* grad_A, int M, int K, int N);
void matmult_backward_B(const float* A, const float* grad_C, float* grad_B, int M, int K, int N);
void pack_strided_to_contiguous(const float* src, float* dst, const int* shape, const int* strides, int ndim, int offset, int total_size);

void reduce_sum_axes(const float* in, float* out, const int* in_shape, int rank, const int* axes, int num_axes, int out_size, int in_size);
void reduce_sum_backward_axes(const float* grad_out, float* grad_in, const int* in_shape, int rank, const int* axes, int num_axes, int in_size);
void reduce_max_axes(const float* in, float* out, int* arg_out, const int* in_shape, int rank, const int* axes, int num_axes, int out_size, int in_size);
void reduce_min_axes(const float* in, float* out, int* arg_out, const int* in_shape, int rank, const int* axes, int num_axes, int out_size, int in_size);
void reduce_scatter_backward(const float* grad_out, float* grad_in, const int* arg_out, int out_size);

void matmul(const float *a, const float *b, float *out, int M, int K, int N);

} // namespace cuda
} // namespace core
} // namespace cvmml

#endif // CVMML_CORE_CUDA_KERNELS_CUH