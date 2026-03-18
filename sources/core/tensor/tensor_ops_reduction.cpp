#include "core/cuda_kernels.cuh"
#include "core/tensor.hpp"
#include "core/tensor_detail.hpp"

#include <cstdint>
#include <limits>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

bool should_parallel_reduction(int64_t size)
{
	return size >= 100000;
}

}

namespace cvmml {
namespace core {

Tensor Tensor::sum(const std::vector<int>& axes, bool keepdim) const
{
	// What: Reduce tensor values with configurable axes and keepdim semantics.
	// Why: Provides a stable reduction contract required by autograd and nn layers.
	Tensor src_cont = this->is_contiguous() ? *this : this->contiguous();
	std::vector<int> norm_axes = detail::normalize_axes(axes, static_cast<int>(shape_.size()));
	std::vector<int> out_shape = detail::reduced_shape_from_axes(shape_, norm_axes, keepdim);

	Tensor result(out_shape);
	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::reduce_sum_axes(src_cont.device_data(), result.device_data(), shape_.data(), static_cast<int>(shape_.size()), norm_axes.data(), static_cast<int>(norm_axes.size()), result.size(), src_cont.size());
	}
	else
	{
		const bool parallel_sum = should_parallel_reduction(src_cont.size());
		float* result_ptr = result.data();
		const float* src_ptr = src_cont.data();
		#ifdef _OPENMP
		#pragma omp parallel for if(parallel_sum)
		#endif
		for ( int i = 0; i < src_cont.size(); ++i )
		{
			std::vector<int> in_coords = detail::unravel_index(i, shape_);
			std::vector<int> keepdim_coords = detail::to_keepdim_coords(in_coords, norm_axes);
			std::vector<int> out_coords = keepdim ? keepdim_coords : detail::to_reduced_coords(keepdim_coords, norm_axes);
			int out_linear = static_cast<int>(detail::ravel_index(out_coords, out_shape));
			#ifdef _OPENMP
			#pragma omp atomic update
			#endif
			result_ptr[out_linear] += src_ptr[i];
		}
	}

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parent = *this;
		std::vector<int> parent_shape = shape_;
		std::vector<int> axes_local = norm_axes;
		std::vector<int> out_shape_local = out_shape;
		result.parents_ = {parent};
		result.backward_fn_ = [parent, result, parent_shape, axes_local, out_shape_local, keepdim]() mutable
		{
			if ( !parent.requires_grad_ )
				return;

			if ( parent.device() == Device::CUDA )
			{
				cuda::reduce_sum_backward_axes(result.device_grad(), parent.device_grad(), parent_shape.data(), static_cast<int>(parent_shape.size()), axes_local.data(), static_cast<int>(axes_local.size()), parent.size());
				return;
			}

			float* grad_out = result.grad();
			float* grad_parent = parent.grad();
			const bool parallel_sum_bwd = should_parallel_reduction(parent.size());
			#ifdef _OPENMP
			#pragma omp parallel for if(parallel_sum_bwd)
			#endif
			for ( int i = 0; i < parent.size(); ++i )
			{
				std::vector<int> in_coords = detail::unravel_index(i, parent_shape);
				std::vector<int> keepdim_coords = detail::to_keepdim_coords(in_coords, axes_local);
				std::vector<int> out_coords = keepdim ? keepdim_coords : detail::to_reduced_coords(keepdim_coords, axes_local);
				int out_linear = static_cast<int>(detail::ravel_index(out_coords, out_shape_local));
				grad_parent[i] += grad_out[out_linear];
			}
		};
	}
	return result;
}

Tensor Tensor::max(const std::vector<int>& axes, bool keepdim) const
{
	// What: Reduce by maximum with first-index tie-break.
	// Why: Deterministic ties are required for stable backward tests.
	Tensor src_cont = this->is_contiguous() ? *this : this->contiguous();
	std::vector<int> norm_axes = detail::normalize_axes(axes, static_cast<int>(shape_.size()));
	std::vector<int> out_shape = detail::reduced_shape_from_axes(shape_, norm_axes, keepdim);

	Tensor result(out_shape);
	std::vector<int> argmax;
	std::shared_ptr<int> argmax_device;

	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		argmax_device = std::shared_ptr<int>(cuda::allocate_int_memory(result.size()), [](int* ptr) { cuda::free_int_memory(ptr); });
		cuda::reduce_max_axes(src_cont.device_data(), result.device_data(), argmax_device.get(), shape_.data(), static_cast<int>(shape_.size()), norm_axes.data(), static_cast<int>(norm_axes.size()), result.size(), src_cont.size());
	}
	else
	{
		argmax.assign(result.size(), -1);
		for ( int i = 0; i < result.size(); ++i )
			result.data()[i] = -std::numeric_limits<float>::infinity();
		const bool parallel_max = should_parallel_reduction(src_cont.size());

		#ifdef _OPENMP
		std::vector<omp_lock_t> locks(result.size());
		for ( int i = 0; i < result.size(); ++i )
			omp_init_lock(&locks[i]);
		#pragma omp parallel for if(parallel_max)
		#endif

		for ( int i = 0; i < src_cont.size(); ++i )
		{
			std::vector<int> in_coords = detail::unravel_index(i, shape_);
			std::vector<int> keepdim_coords = detail::to_keepdim_coords(in_coords, norm_axes);
			std::vector<int> out_coords = keepdim ? keepdim_coords : detail::to_reduced_coords(keepdim_coords, norm_axes);
			int out_linear = static_cast<int>(detail::ravel_index(out_coords, out_shape));
			float value = src_cont.data()[i];

			#ifdef _OPENMP
			if ( parallel_max )
				omp_set_lock(&locks[out_linear]);
			#endif
			if ( value > result.data()[out_linear] || (value == result.data()[out_linear] && (argmax[out_linear] < 0 || i < argmax[out_linear])) )
			{
				result.data()[out_linear] = value;
				argmax[out_linear] = i;
			}
			#ifdef _OPENMP
			if ( parallel_max )
				omp_unset_lock(&locks[out_linear]);
			#endif
		}

		#ifdef _OPENMP
		for ( int i = 0; i < result.size(); ++i )
			omp_destroy_lock(&locks[i]);
		#endif
	}

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parent = *this;
		std::vector<int> argmax_local = argmax;
		result.parents_ = {parent};
		result.backward_fn_ = [parent, result, argmax_local, argmax_device]() mutable
		{
			if ( !parent.requires_grad_ )
				return;

			if ( parent.device() == Device::CUDA )
			{
				cuda::reduce_scatter_backward(result.device_grad(), parent.device_grad(), argmax_device.get(), result.size());
				return;
			}

			float* grad_out = result.grad();
			float* grad_parent = parent.grad();
			const bool parallel_max_bwd = should_parallel_reduction(static_cast<int64_t>(argmax_local.size()));
			#ifdef _OPENMP
			#pragma omp parallel for if(parallel_max_bwd)
			#endif
			for ( int i = 0; i < static_cast<int>(argmax_local.size()); ++i )
			{
				int src_idx = argmax_local[i];
				if ( src_idx >= 0 )
					grad_parent[src_idx] += grad_out[i];
			}
		};
	}
	return result;
}

Tensor Tensor::min(const std::vector<int>& axes, bool keepdim) const
{
	// What: Reduce by minimum with first-index tie-break, aligned with max policy.
	// Why: Keeps deterministic gradient routing and API symmetry.
	Tensor src_cont = this->is_contiguous() ? *this : this->contiguous();
	std::vector<int> norm_axes = detail::normalize_axes(axes, static_cast<int>(shape_.size()));
	std::vector<int> out_shape = detail::reduced_shape_from_axes(shape_, norm_axes, keepdim);

	Tensor result(out_shape);
	std::vector<int> argmin;
	std::shared_ptr<int> argmin_device;

	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		argmin_device = std::shared_ptr<int>(cuda::allocate_int_memory(result.size()), [](int* ptr) { cuda::free_int_memory(ptr); });
		cuda::reduce_min_axes(src_cont.device_data(), result.device_data(), argmin_device.get(), shape_.data(), static_cast<int>(shape_.size()), norm_axes.data(), static_cast<int>(norm_axes.size()), result.size(), src_cont.size());
	}
	else
	{
		argmin.assign(result.size(), -1);
		for ( int i = 0; i < result.size(); ++i )
			result.data()[i] = std::numeric_limits<float>::infinity();
		const bool parallel_min = should_parallel_reduction(src_cont.size());

		#ifdef _OPENMP
		std::vector<omp_lock_t> locks(result.size());
		for ( int i = 0; i < result.size(); ++i )
			omp_init_lock(&locks[i]);
		#pragma omp parallel for if(parallel_min)
		#endif

		for ( int i = 0; i < src_cont.size(); ++i )
		{
			std::vector<int> in_coords = detail::unravel_index(i, shape_);
			std::vector<int> keepdim_coords = detail::to_keepdim_coords(in_coords, norm_axes);
			std::vector<int> out_coords = keepdim ? keepdim_coords : detail::to_reduced_coords(keepdim_coords, norm_axes);
			int out_linear = static_cast<int>(detail::ravel_index(out_coords, out_shape));
			float value = src_cont.data()[i];

			#ifdef _OPENMP
			if ( parallel_min )
				omp_set_lock(&locks[out_linear]);
			#endif
			if ( value < result.data()[out_linear] || (value == result.data()[out_linear] && (argmin[out_linear] < 0 || i < argmin[out_linear])) )
			{
				result.data()[out_linear] = value;
				argmin[out_linear] = i;
			}
			#ifdef _OPENMP
			if ( parallel_min )
				omp_unset_lock(&locks[out_linear]);
			#endif
		}

		#ifdef _OPENMP
		for ( int i = 0; i < result.size(); ++i )
			omp_destroy_lock(&locks[i]);
		#endif
	}

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parent = *this;
		std::vector<int> argmin_local = argmin;
		result.parents_ = {parent};
		result.backward_fn_ = [parent, result, argmin_local, argmin_device]() mutable
		{
			if ( !parent.requires_grad_ )
				return;

			if ( parent.device() == Device::CUDA )
			{
				cuda::reduce_scatter_backward(result.device_grad(), parent.device_grad(), argmin_device.get(), result.size());
				return;
			}

			float* grad_out = result.grad();
			float* grad_parent = parent.grad();
			const bool parallel_min_bwd = should_parallel_reduction(static_cast<int64_t>(argmin_local.size()));
			#ifdef _OPENMP
			#pragma omp parallel for if(parallel_min_bwd)
			#endif
			for ( int i = 0; i < static_cast<int>(argmin_local.size()); ++i )
			{
				int src_idx = argmin_local[i];
				if ( src_idx >= 0 )
					grad_parent[src_idx] += grad_out[i];
			}
		};
	}
	return result;
}

} // namespace core
} // namespace cvmml
