#include "core/cuda_kernels.cuh"
#include "core/tensor.hpp"
#include "core/tensor_detail.hpp"

#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

bool should_parallel_matmul(int64_t batch_count, int M, int K, int N)
{
	const int64_t work = batch_count * static_cast<int64_t>(M) * static_cast<int64_t>(K) * static_cast<int64_t>(N);
	return work >= 500000;
}

}

namespace cvmml {
namespace core {

Tensor Tensor::matmult(const Tensor& rhs) const
{
	// What: Batched matrix multiplication with broadcasted batch dimensions.
	// Why: Keeps matmul API stable while supporting CPU and CUDA paths.
	if ( this->shape_.size() < 2 || rhs.shape().size() < 2 )
		throw std::invalid_argument("matmult requires tensors with at least 2 dimensions.");
	detail::check_same_device(*this, rhs);

	int M = this->shape_[this->shape_.size() - 2];
	int K = this->shape_[this->shape_.size() - 1];
	int K2 = rhs.shape()[rhs.shape().size() - 2];
	int N = rhs.shape()[rhs.shape().size() - 1];
	if ( K != K2 )
		throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");

	if ( (!this->is_contiguous() && this->requires_grad_) || (!rhs.is_contiguous() && rhs.requires_grad_) )
		throw std::invalid_argument("matmult backward on non-contiguous views is not supported yet. Call contiguous() before matmult when gradients are required.");

	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

	std::vector<int> lhs_batch_shape(this->shape_.begin(), this->shape_.end() - 2);
	std::vector<int> rhs_batch_shape(rhs.shape().begin(), rhs.shape().end() - 2);
	std::vector<int> out_batch_shape = detail::broadcast_batch_shape(lhs_batch_shape, rhs_batch_shape);

	std::vector<int> out_shape = out_batch_shape;
	out_shape.push_back(M);
	out_shape.push_back(N);

	int64_t batch_count = out_batch_shape.empty() ? 1 : detail::product_of(out_batch_shape);
	int64_t lhs_matrix_size = static_cast<int64_t>(M) * K;
	int64_t rhs_matrix_size = static_cast<int64_t>(K) * N;
	int64_t out_matrix_size = static_cast<int64_t>(M) * N;

	Tensor result(out_shape);
	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		for ( int64_t b = 0; b < batch_count; ++b )
		{
			std::vector<int> out_batch_idx = detail::unravel_index(b, out_batch_shape);
			int64_t lhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
			int64_t rhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);

			const float* A_ptr = lhs_cont.device_data() + lhs_batch_linear * lhs_matrix_size;
			const float* B_ptr = rhs_cont.device_data() + rhs_batch_linear * rhs_matrix_size;
			float* C_ptr = result.device_data() + b * out_matrix_size;
			cuda::matmul(A_ptr, B_ptr, C_ptr, M, K, N);
		}
	}
	else
	{
		const bool parallel_forward = should_parallel_matmul(batch_count, M, K, N);
		for ( int64_t b = 0; b < batch_count; ++b )
		{
			std::vector<int> out_batch_idx = detail::unravel_index(b, out_batch_shape);
			int64_t lhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
			int64_t rhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);

			const float* A_ptr = lhs_cont.data() + lhs_batch_linear * lhs_matrix_size;
			const float* B_ptr = rhs_cont.data() + rhs_batch_linear * rhs_matrix_size;
			float* C_ptr = result.data() + b * out_matrix_size;

			#ifdef _OPENMP
			#pragma omp parallel for if(parallel_forward)
			#endif
			for ( int i = 0; i < M; ++i )
				for ( int j = 0; j < N; ++j )
				{
					float sum = 0.0f;
					for ( int k = 0; k < K; ++k )
						sum += A_ptr[i * K + k] * B_ptr[k * N + j];
					C_ptr[i * N + j] = sum;
				}
		}
	}

	result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		Tensor parentB = rhs;
		result.parents_ = {parentA, parentB};
		result.backward_fn_ = [parentA, parentB, result, M, K, N, lhs_batch_shape, rhs_batch_shape, out_batch_shape]() mutable
		{
			int64_t batch_count_local = out_batch_shape.empty() ? 1 : detail::product_of(out_batch_shape);
			int64_t lhs_matrix_size_local = static_cast<int64_t>(M) * K;
			int64_t rhs_matrix_size_local = static_cast<int64_t>(K) * N;
			int64_t out_matrix_size_local = static_cast<int64_t>(M) * N;
			const bool parallel_backward = should_parallel_matmul(batch_count_local, M, K, N);

			if ( parentA.requires_grad_ )
			{
				if ( parentA.device() == Device::CUDA )
				{
					for ( int64_t b = 0; b < batch_count_local; ++b )
					{
						std::vector<int> out_batch_idx = detail::unravel_index(b, out_batch_shape);
						int64_t rhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);
						int64_t lhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
						cuda::matmult_backward_A(result.device_grad() + b * out_matrix_size_local, parentB.device_data() + rhs_batch_linear * rhs_matrix_size_local, parentA.device_grad() + lhs_batch_linear * lhs_matrix_size_local, M, K, N);
					}
				}
				else
				{
					for ( int64_t b = 0; b < batch_count_local; ++b )
					{
						std::vector<int> out_batch_idx = detail::unravel_index(b, out_batch_shape);
						int64_t rhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);
						int64_t lhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
						const float* gradC = result.grad() + b * out_matrix_size_local;
						const float* B_ptr = parentB.data() + rhs_batch_linear * rhs_matrix_size_local;
						float* gradA = parentA.grad() + lhs_batch_linear * lhs_matrix_size_local;
						#ifdef _OPENMP
						#pragma omp parallel for if(parallel_backward)
						#endif
						for ( int i = 0; i < M; ++i )
							for ( int j = 0; j < N; ++j )
							{
								float dC = gradC[i * N + j];
								for ( int k = 0; k < K; ++k )
									gradA[i * K + k] += dC * B_ptr[k * N + j];
							}
					}
				}
			}

			if ( parentB.requires_grad_ )
			{
				if ( parentB.device() == Device::CUDA )
				{
					for ( int64_t b = 0; b < batch_count_local; ++b )
					{
						std::vector<int> out_batch_idx = detail::unravel_index(b, out_batch_shape);
						int64_t lhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
						int64_t rhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);
						cuda::matmult_backward_B(parentA.device_data() + lhs_batch_linear * lhs_matrix_size_local, result.device_grad() + b * out_matrix_size_local, parentB.device_grad() + rhs_batch_linear * rhs_matrix_size_local, M, K, N);
					}
				}
				else
				{
					for ( int64_t b = 0; b < batch_count_local; ++b )
					{
						std::vector<int> out_batch_idx = detail::unravel_index(b, out_batch_shape);
						int64_t lhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
						int64_t rhs_batch_linear = detail::map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);
						const float* A_ptr = parentA.data() + lhs_batch_linear * lhs_matrix_size_local;
						const float* gradC = result.grad() + b * out_matrix_size_local;
						float* gradB = parentB.grad() + rhs_batch_linear * rhs_matrix_size_local;
						#ifdef _OPENMP
						#pragma omp parallel for collapse(2) if(parallel_backward)
						#endif
						for ( int k = 0; k < K; ++k )
							for ( int j = 0; j < N; ++j )
							{
								float sum = 0.0f;
								for ( int i = 0; i < M; ++i )
									sum += A_ptr[i * K + k] * gradC[i * N + j];
								gradB[k * N + j] += sum;
							}
					}
				}
			}
		};
	}

	return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const
{
	// What: Create a stride-based transpose view over two dimensions.
	// Why: Avoids data copies while keeping backward mapping explicit.
	int rank = static_cast<int>(shape_.size());
	if ( rank < 2 )
		throw std::invalid_argument("transpose requires tensors with at least 2 dimensions.");
	if ( dim0 < 0 )
		dim0 += rank;
	if ( dim1 < 0 )
		dim1 += rank;
	if ( dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank || dim0 == dim1 )
		throw std::invalid_argument("Invalid transpose dimensions.");

	Tensor tr = *this;
	std::swap(tr.shape_[dim0], tr.shape_[dim1]);
	std::swap(tr.strides_[dim0], tr.strides_[dim1]);
	tr.is_view_ = true;
	tr.grad_.reset();
	tr.device_grad_.reset();
	tr.parents_.clear();
	tr.backward_fn_ = std::function<void()>();

	tr.set_requires_grad(this->requires_grad_);
	if ( tr.requires_grad_ )
	{
		Tensor parent = *this;
		std::vector<int> parent_shape = shape_;
		tr.parents_ = {parent};
		tr.backward_fn_ = [parent, tr, parent_shape, dim0, dim1]() mutable
		{
			if ( !parent.requires_grad_ )
				return;

			if ( parent.device() == Device::CUDA )
			{
				cuda::add_transpose_nd(tr.device_grad(), parent.device_grad(), parent_shape.data(), static_cast<int>(parent_shape.size()), dim0, dim1, parent.size());
				return;
			}

			int total = parent.size();
			for ( int linear_parent = 0; linear_parent < total; ++linear_parent )
			{
				std::vector<int> parent_coords = detail::unravel_index(linear_parent, parent_shape);
				std::swap(parent_coords[dim0], parent_coords[dim1]);
				std::vector<int> y_shape = parent_shape;
				std::swap(y_shape[dim0], y_shape[dim1]);
				int linear_y = static_cast<int>(detail::ravel_index(parent_coords, y_shape));
				parent.grad()[linear_parent] += tr.grad()[linear_y];
			}
		};
	}
	return tr;
}

} // namespace core
} // namespace cvmml
