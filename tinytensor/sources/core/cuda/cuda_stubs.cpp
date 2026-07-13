#include "cuda_kernels.cuh"
#include "tensor.hpp"

#include <stdexcept>

namespace tinytensor {
namespace core {

namespace {

[[noreturn]] void throw_cuda_unavailable()
{
    throw std::runtime_error(
        "TinyMML was built without CUDA support. Rebuild with './build.sh --cuda'."
    );
}

} // namespace

Device Tensor::device() const { return device_; }
float* Tensor::device_data() const { return device_data_.get(); }
Tensor Tensor::to_cuda() const { throw_cuda_unavailable(); }
Tensor Tensor::to_cpu() const { return *this; }

namespace cuda {

float* allocate_memory(int) { throw_cuda_unavailable(); }
int* allocate_int_memory(int) { throw_cuda_unavailable(); }
void free_memory(float*) { throw_cuda_unavailable(); }
void free_int_memory(int*) { throw_cuda_unavailable(); }
void set_memory(float*, float, int) { throw_cuda_unavailable(); }
void copy_to_device(float*, const float*, int) { throw_cuda_unavailable(); }
void copy_to_host(float*, const float*, int) { throw_cuda_unavailable(); }
void fill_ones(float*, int) { throw_cuda_unavailable(); }

void add_arrays(const float*, const float*, float*, int) { throw_cuda_unavailable(); }
void sub_arrays(const float*, const float*, float*, int) { throw_cuda_unavailable(); }
void mul_arrays(const float*, const float*, float*, int) { throw_cuda_unavailable(); }
void div_arrays(const float*, const float*, float*, int) { throw_cuda_unavailable(); }
void exp_array(const float*, float*, int) { throw_cuda_unavailable(); }
void log_array(const float*, float*, int) { throw_cuda_unavailable(); }
void relu_array(const float*, float*, int) { throw_cuda_unavailable(); }
void relu_backward_array(float*, const float*, const float*, int) { throw_cuda_unavailable(); }
void sqrt_array(const float*, float*, int) { throw_cuda_unavailable(); }
void sqrt_backward_array(float*, const float*, const float*, int) { throw_cuda_unavailable(); }
void add_scalar(const float*, float, float*, int) { throw_cuda_unavailable(); }
void sub_scalar(const float*, float, float*, int) { throw_cuda_unavailable(); }
void mul_scalar(const float*, float, float*, int) { throw_cuda_unavailable(); }
void div_scalar(const float*, float, float*, int) { throw_cuda_unavailable(); }

void add_mul_arrays(float*, const float*, const float*, int) { throw_cuda_unavailable(); }
void add_div_arrays(float*, const float*, const float*, int) { throw_cuda_unavailable(); }
void sub_mul_div_sqr_arrays(float*, const float*, const float*, const float*, int) { throw_cuda_unavailable(); }
void add_mul_scalar_arrays(float*, const float*, float, int) { throw_cuda_unavailable(); }
void add_div_scalar_arrays(float*, const float*, float, int) { throw_cuda_unavailable(); }
void add_transpose_nd(const float*, float*, const int*, int, int, int, int) { throw_cuda_unavailable(); }
void matmult_backward_A(const float*, const float*, float*, int, int, int) { throw_cuda_unavailable(); }
void matmult_backward_B(const float*, const float*, float*, int, int, int) { throw_cuda_unavailable(); }
void pack_strided_to_contiguous(const float*, float*, const int*, const int*, int, int, int) { throw_cuda_unavailable(); }

void reduce_sum_axes(const float*, float*, const int*, int, const int*, int, int, int) { throw_cuda_unavailable(); }
void reduce_sum_backward_axes(const float*, float*, const int*, int, const int*, int, int) { throw_cuda_unavailable(); }
void reduce_max_axes(const float*, float*, int*, const int*, int, const int*, int, int, int) { throw_cuda_unavailable(); }
void reduce_min_axes(const float*, float*, int*, const int*, int, const int*, int, int, int) { throw_cuda_unavailable(); }
void reduce_scatter_backward(const float*, float*, const int*, int) { throw_cuda_unavailable(); }
void matmul(const float*, const float*, float*, int, int, int) { throw_cuda_unavailable(); }
void im2col(const float*, float*, int, int, int, int, int, int, int, int, int, int) { throw_cuda_unavailable(); }

} // namespace cuda
} // namespace core
} // namespace tinytensor
