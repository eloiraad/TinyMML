#include "cuda_kernels.cuh"
#include "tensor.hpp"
#include "tensor_detail.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tinytensor {
namespace core {

Tensor Tensor::operator+(const Tensor& rhs) const
{
	detail::check_same_device(*this, rhs);
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

	std::vector<int> out_shape = detail::broadcast_shape_nd(lhs_cont.shape(), rhs_cont.shape());
	Tensor result(out_shape);

	if ( this->device_ == Device::CUDA && lhs_cont.shape() == rhs_cont.shape() && out_shape == lhs_cont.shape() )
	{
		result = result.to_cuda();
		cuda::add_arrays(lhs_cont.device_data(), rhs_cont.device_data(), result.device_data(), result.size());
	}
	else
	{
		Tensor lhs_host = (this->device_ == Device::CUDA) ? lhs_cont.to_cpu() : lhs_cont;
		Tensor rhs_host = (this->device_ == Device::CUDA) ? rhs_cont.to_cpu() : rhs_cont;

		for ( int i = 0; i < result.size(); ++i )
		{
			std::vector<int> out_coords = detail::unravel_index(i, out_shape);
			int64_t ia = detail::linear_for_broadcast_operand(out_coords, lhs_host.shape(), lhs_host.strides());
			int64_t ib = detail::linear_for_broadcast_operand(out_coords, rhs_host.shape(), rhs_host.strides());
			result.data()[i] = lhs_host.data()[ia] + rhs_host.data()[ib];
		}
		if ( this->device_ == Device::CUDA )
			result = result.to_cuda();
	}

	result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		Tensor parentB = rhs;
		std::vector<int> out_shape_local = out_shape;
		result.parents_ = {parentA, parentB};
		result.backward_fn_ = [out_shape_local](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			const Tensor& parentB = result.parents_[1];
			float* grad_out = result.grad();
			int out_size = static_cast<int>(detail::product_of(out_shape_local));
			if ( parentA.requires_grad_ )
			{
				float* grad_a = parentA.grad();
				for ( int i = 0; i < out_size; ++i )
				{
					std::vector<int> out_coords = detail::unravel_index(i, out_shape_local);
					int64_t ia = detail::linear_for_broadcast_operand(out_coords, parentA.shape(), detail::make_contiguous_strides(parentA.shape()));
					grad_a[ia] += grad_out[i];
				}
				if ( parentA.device() == Device::CUDA )
					cuda::copy_to_device(parentA.device_grad(), grad_a, parentA.size());
			}
			if ( parentB.requires_grad_ )
			{
				float* grad_b = parentB.grad();
				for ( int i = 0; i < out_size; ++i )
				{
					std::vector<int> out_coords = detail::unravel_index(i, out_shape_local);
					int64_t ib = detail::linear_for_broadcast_operand(out_coords, parentB.shape(), detail::make_contiguous_strides(parentB.shape()));
					grad_b[ib] += grad_out[i];
				}
				if ( parentB.device() == Device::CUDA )
					cuda::copy_to_device(parentB.device_grad(), grad_b, parentB.size());
			}
		};
	}
	return result;
}

Tensor Tensor::operator-(const Tensor& rhs) const
{
	detail::check_same_device(*this, rhs);
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

	std::vector<int> out_shape = detail::broadcast_shape_nd(lhs_cont.shape(), rhs_cont.shape());
	Tensor result(out_shape);

	if ( this->device_ == Device::CUDA && lhs_cont.shape() == rhs_cont.shape() && out_shape == lhs_cont.shape() )
	{
		result = result.to_cuda();
		cuda::sub_arrays(lhs_cont.device_data(), rhs_cont.device_data(), result.device_data(), result.size());
	}
	else
	{
		Tensor lhs_host = (this->device_ == Device::CUDA) ? lhs_cont.to_cpu() : lhs_cont;
		Tensor rhs_host = (this->device_ == Device::CUDA) ? rhs_cont.to_cpu() : rhs_cont;
		for ( int i = 0; i < result.size(); ++i )
		{
			std::vector<int> out_coords = detail::unravel_index(i, out_shape);
			int64_t ia = detail::linear_for_broadcast_operand(out_coords, lhs_host.shape(), lhs_host.strides());
			int64_t ib = detail::linear_for_broadcast_operand(out_coords, rhs_host.shape(), rhs_host.strides());
			result.data()[i] = lhs_host.data()[ia] - rhs_host.data()[ib];
		}
		if ( this->device_ == Device::CUDA )
			result = result.to_cuda();
	}

	result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		Tensor parentB = rhs;
		std::vector<int> out_shape_local = out_shape;
		result.parents_ = {parentA, parentB};
		result.backward_fn_ = [out_shape_local](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			const Tensor& parentB = result.parents_[1];
			float* grad_out = result.grad();
			int out_size = static_cast<int>(detail::product_of(out_shape_local));
			if ( parentA.requires_grad_ )
			{
				float* grad_a = parentA.grad();
				for ( int i = 0; i < out_size; ++i )
				{
					std::vector<int> out_coords = detail::unravel_index(i, out_shape_local);
					int64_t ia = detail::linear_for_broadcast_operand(out_coords, parentA.shape(), detail::make_contiguous_strides(parentA.shape()));
					grad_a[ia] += grad_out[i];
				}
				if ( parentA.device() == Device::CUDA )
					cuda::copy_to_device(parentA.device_grad(), grad_a, parentA.size());
			}
			if ( parentB.requires_grad_ )
			{
				float* grad_b = parentB.grad();
				for ( int i = 0; i < out_size; ++i )
				{
					std::vector<int> out_coords = detail::unravel_index(i, out_shape_local);
					int64_t ib = detail::linear_for_broadcast_operand(out_coords, parentB.shape(), detail::make_contiguous_strides(parentB.shape()));
					grad_b[ib] -= grad_out[i];
				}
				if ( parentB.device() == Device::CUDA )
					cuda::copy_to_device(parentB.device_grad(), grad_b, parentB.size());
			}
		};
	}
	return result;
}

Tensor Tensor::operator*(const Tensor& rhs) const
{
	detail::check_same_device(*this, rhs);
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

	std::vector<int> out_shape = detail::broadcast_shape_nd(lhs_cont.shape(), rhs_cont.shape());
	Tensor result(out_shape);

	if ( this->device_ == Device::CUDA && lhs_cont.shape() == rhs_cont.shape() && out_shape == lhs_cont.shape() )
	{
		result = result.to_cuda();
		cuda::mul_arrays(lhs_cont.device_data(), rhs_cont.device_data(), result.device_data(), result.size());
	}
	else
	{
		Tensor lhs_host = (this->device_ == Device::CUDA) ? lhs_cont.to_cpu() : lhs_cont;
		Tensor rhs_host = (this->device_ == Device::CUDA) ? rhs_cont.to_cpu() : rhs_cont;
		for ( int i = 0; i < result.size(); ++i )
		{
			std::vector<int> out_coords = detail::unravel_index(i, out_shape);
			int64_t ia = detail::linear_for_broadcast_operand(out_coords, lhs_host.shape(), lhs_host.strides());
			int64_t ib = detail::linear_for_broadcast_operand(out_coords, rhs_host.shape(), rhs_host.strides());
			result.data()[i] = lhs_host.data()[ia] * rhs_host.data()[ib];
		}
		if ( this->device_ == Device::CUDA )
			result = result.to_cuda();
	}

	result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		Tensor parentB = rhs;
		std::vector<int> out_shape_local = out_shape;
		result.parents_ = {parentA, parentB};
		result.backward_fn_ = [out_shape_local](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			const Tensor& parentB = result.parents_[1];
			Tensor pA_host = (parentA.device() == Device::CUDA) ? parentA.to_cpu() : parentA;
			Tensor pB_host = (parentB.device() == Device::CUDA) ? parentB.to_cpu() : parentB;
			float* grad_out = result.grad();
			int out_size = static_cast<int>(detail::product_of(out_shape_local));
			if ( parentA.requires_grad_ )
			{
				float* grad_a = parentA.grad();
				for ( int i = 0; i < out_size; ++i )
				{
					std::vector<int> out_coords = detail::unravel_index(i, out_shape_local);
					int64_t ia = detail::linear_for_broadcast_operand(out_coords, parentA.shape(), detail::make_contiguous_strides(parentA.shape()));
					int64_t ib = detail::linear_for_broadcast_operand(out_coords, pB_host.shape(), pB_host.strides());
					grad_a[ia] += grad_out[i] * pB_host.data()[ib];
				}
				if ( parentA.device() == Device::CUDA )
					cuda::copy_to_device(parentA.device_grad(), grad_a, parentA.size());
			}
			if ( parentB.requires_grad_ )
			{
				float* grad_b = parentB.grad();
				for ( int i = 0; i < out_size; ++i )
				{
					std::vector<int> out_coords = detail::unravel_index(i, out_shape_local);
					int64_t ib = detail::linear_for_broadcast_operand(out_coords, parentB.shape(), detail::make_contiguous_strides(parentB.shape()));
					int64_t ia = detail::linear_for_broadcast_operand(out_coords, pA_host.shape(), pA_host.strides());
					grad_b[ib] += grad_out[i] * pA_host.data()[ia];
				}
				if ( parentB.device() == Device::CUDA )
					cuda::copy_to_device(parentB.device_grad(), grad_b, parentB.size());
			}
		};
	}
	return result;
}

Tensor Tensor::operator/(const Tensor& rhs) const
{
	detail::check_same_device(*this, rhs);
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

	std::vector<int> out_shape = detail::broadcast_shape_nd(lhs_cont.shape(), rhs_cont.shape());
	Tensor result(out_shape);

	if ( this->device_ == Device::CUDA && lhs_cont.shape() == rhs_cont.shape() && out_shape == lhs_cont.shape() )
	{
		result = result.to_cuda();
		cuda::div_arrays(lhs_cont.device_data(), rhs_cont.device_data(), result.device_data(), result.size());
	}
	else
	{
		Tensor lhs_host = (this->device_ == Device::CUDA) ? lhs_cont.to_cpu() : lhs_cont;
		Tensor rhs_host = (this->device_ == Device::CUDA) ? rhs_cont.to_cpu() : rhs_cont;
		for ( int i = 0; i < result.size(); ++i )
		{
			std::vector<int> out_coords = detail::unravel_index(i, out_shape);
			int64_t ia = detail::linear_for_broadcast_operand(out_coords, lhs_host.shape(), lhs_host.strides());
			int64_t ib = detail::linear_for_broadcast_operand(out_coords, rhs_host.shape(), rhs_host.strides());
			result.data()[i] = lhs_host.data()[ia] / rhs_host.data()[ib];
		}
		if ( this->device_ == Device::CUDA )
			result = result.to_cuda();
	}

	result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		Tensor parentB = rhs;
		std::vector<int> out_shape_local = out_shape;
		result.parents_ = {parentA, parentB};
		result.backward_fn_ = [out_shape_local](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			const Tensor& parentB = result.parents_[1];
			Tensor pA_host = (parentA.device() == Device::CUDA) ? parentA.to_cpu() : parentA;
			Tensor pB_host = (parentB.device() == Device::CUDA) ? parentB.to_cpu() : parentB;
			float* grad_out = result.grad();
			int out_size = static_cast<int>(detail::product_of(out_shape_local));
			if ( parentA.requires_grad_ )
			{
				float* grad_a = parentA.grad();
				for ( int i = 0; i < out_size; ++i )
				{
					std::vector<int> out_coords = detail::unravel_index(i, out_shape_local);
					int64_t ia = detail::linear_for_broadcast_operand(out_coords, parentA.shape(), detail::make_contiguous_strides(parentA.shape()));
					int64_t ib = detail::linear_for_broadcast_operand(out_coords, pB_host.shape(), pB_host.strides());
					grad_a[ia] += grad_out[i] / pB_host.data()[ib];
				}
				if ( parentA.device() == Device::CUDA )
					cuda::copy_to_device(parentA.device_grad(), grad_a, parentA.size());
			}
			if ( parentB.requires_grad_ )
			{
				float* grad_b = parentB.grad();
				for ( int i = 0; i < out_size; ++i )
				{
					std::vector<int> out_coords = detail::unravel_index(i, out_shape_local);
					int64_t ib = detail::linear_for_broadcast_operand(out_coords, parentB.shape(), detail::make_contiguous_strides(parentB.shape()));
					int64_t ia = detail::linear_for_broadcast_operand(out_coords, pA_host.shape(), pA_host.strides());
					float denom = pB_host.data()[ib] * pB_host.data()[ib];
					grad_b[ib] += grad_out[i] * (-pA_host.data()[ia] / denom);
				}
				if ( parentB.device() == Device::CUDA )
					cuda::copy_to_device(parentB.device_grad(), grad_b, parentB.size());
			}
		};
	}
	return result;
}

Tensor Tensor::operator+(float scalar) const
{
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor result(this->shape_);
	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::add_scalar(lhs_cont.device_data(), scalar, result.device_data(), total_size_);
	}
	else
		for ( int i = 0; i < this->total_size_; ++i )
			result.data()[i] = lhs_cont.data()[i] + scalar;

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		result.parents_ = {parentA};
		result.backward_fn_ = [](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			if ( parentA.requires_grad_ )
			{
				if ( parentA.device() == Device::CUDA )
					cuda::add_arrays(parentA.device_grad(), result.device_grad(), parentA.device_grad(), parentA.size());
				else
					for ( int i = 0; i < parentA.size(); ++i )
						parentA.grad()[i] += result.grad()[i];
			}
		};
	}
	return result;
}

Tensor Tensor::operator-(float scalar) const
{
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor result(this->shape_);
	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::sub_scalar(lhs_cont.device_data(), scalar, result.device_data(), total_size_);
	}
	else
		for ( int i = 0; i < this->total_size_; ++i )
			result.data()[i] = lhs_cont.data()[i] - scalar;

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		result.parents_ = {parentA};
		result.backward_fn_ = [](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			if ( parentA.requires_grad_ )
			{
				if ( parentA.device() == Device::CUDA )
					cuda::add_arrays(parentA.device_grad(), result.device_grad(), parentA.device_grad(), parentA.size());
				else
					for ( int i = 0; i < parentA.size(); ++i )
						parentA.grad()[i] += result.grad()[i];
			}
		};
	}
	return result;
}

Tensor Tensor::operator*(float scalar) const
{
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor result(this->shape_);
	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::mul_scalar(lhs_cont.device_data(), scalar, result.device_data(), total_size_);
	}
	else
		for ( int i = 0; i < this->total_size_; ++i )
			result.data()[i] = lhs_cont.data()[i] * scalar;

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		result.parents_ = {parentA};
		result.backward_fn_ = [scalar](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			if ( parentA.requires_grad_ )
			{
				if ( parentA.device() == Device::CUDA )
					cuda::add_mul_scalar_arrays(parentA.device_grad(), result.device_grad(), scalar, parentA.size());
				else
					for ( int i = 0; i < parentA.size(); ++i )
						parentA.grad()[i] += result.grad()[i] * scalar;
			}
		};
	}
	return result;
}

Tensor Tensor::operator/(float scalar) const
{
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor result(this->shape_);
	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::div_scalar(lhs_cont.device_data(), scalar, result.device_data(), total_size_);
	}
	else
		for ( int i = 0; i < this->total_size_; ++i )
			result.data()[i] = lhs_cont.data()[i] / scalar;

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		result.parents_ = {parentA};
		result.backward_fn_ = [scalar](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			if ( parentA.requires_grad_ )
			{
				if ( parentA.device() == Device::CUDA )
					cuda::add_div_scalar_arrays(parentA.device_grad(), result.device_grad(), scalar, parentA.size());
				else
					for ( int i = 0; i < parentA.size(); ++i )
						parentA.grad()[i] += result.grad()[i] / scalar;
			}
		};
	}
	return result;
}

Tensor Tensor::exp() const
{
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor result(this->shape_);
	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::exp_array(lhs_cont.device_data(), result.device_data(), total_size_);
	}
	else
		for ( int i = 0; i < this->total_size_; ++i )
			result.data()[i] = std::exp(lhs_cont.data()[i]);

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		result.parents_ = {parentA};
		result.backward_fn_ = [](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			if ( parentA.requires_grad_ )
			{
				if ( parentA.device() == Device::CUDA )
					cuda::add_mul_arrays(parentA.device_grad(), result.device_grad(), result.device_data(), parentA.size());
				else
					for ( int i = 0; i < parentA.size(); ++i )
						parentA.grad()[i] += result.grad()[i] * result.data()[i];
			}
		};
	}
	return result;
}

Tensor Tensor::log() const
{
	Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
	Tensor result(this->shape_);
	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::log_array(lhs_cont.device_data(), result.device_data(), total_size_);
	}
	else
		for ( int i = 0; i < this->total_size_; ++i )
			result.data()[i] = std::log(lhs_cont.data()[i]);

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parentA = *this;
		result.parents_ = {parentA};
		result.backward_fn_ = [](const Tensor& result) mutable
		{
			const Tensor& parentA = result.parents_[0];
			if ( parentA.requires_grad_ )
			{
				if ( parentA.device() == Device::CUDA )
					cuda::add_div_arrays(parentA.device_grad(), result.device_grad(), parentA.device_data(), parentA.size());
				else
					for ( int i = 0; i < parentA.size(); ++i )
						parentA.grad()[i] += result.grad()[i] / parentA.data()[i];
			}
		};
	}
	return result;
}

Tensor Tensor::relu() const
{
	Tensor src = this->is_contiguous() ? *this : this->contiguous();
	Tensor result(this->shape_);

	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::relu_array(src.device_data(), result.device_data(), total_size_);
	}
	else
		for ( int i = 0; i < total_size_; i++ )
			result.data()[i] = std::max(0.0f, src.data()[i]);

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parent = *this;
		result.parents_ = {parent};
		result.backward_fn_ = [](const Tensor& result) mutable
		{
			const Tensor& parent = result.parents_[0];
			if ( !parent.requires_grad_ )
				return; //? Inutile ?
			if ( parent.device() == Device::CUDA )
				cuda::relu_backward_array(parent.device_grad(), result.device_grad(), parent.device_data(), parent.size());
			else
			{
				float* gp = parent.grad();
				float* go = result.grad();
				float* xp = parent.data();
				for ( int i = 0; i < parent.size(); i++ )
					gp[i] += (xp[i] > 0.0f) ? go[i] : 0.0f;
			}
		};
	}
	return result;
}

Tensor Tensor::sqrt() const
{
	Tensor src = this->is_contiguous() ? *this : this->contiguous();
	Tensor result(this->shape_);

	if ( this->device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::sqrt_array(src.device_data(), result.device_data(), total_size_);
	}
	else
		for ( int i = 0; i < total_size_; i++ )
			result.data()[i] = std::sqrt(src.data()[i]); //? sqrtf ?

	result.set_requires_grad(this->requires_grad_);
	if ( result.requires_grad_ )
	{
		Tensor parent = *this;
		result.parents_ = {parent};
		result.backward_fn_ = [](const Tensor& result) mutable
		{
			const Tensor& parent = result.parents_[0];
			if ( !parent.requires_grad_ )
				return;
			if ( parent.device() == Device::CUDA )
				cuda::sqrt_backward_array(parent.device_grad(), result.device_grad(), result.device_data(), parent.size());
			else
			{
				float* gp = parent.grad();
				float* go = result.grad();
				float* y = result.data();
				for ( int i = 0; i < parent.size(); i++ )
					if ( y[i] > 0.0f )
						gp[i] += go[i] * (0.5f / y[i]);
			}
		};
	}
	return result;
}

void Tensor::subtract_( const Tensor& rhs )
{
	detail::check_same_device(*this, rhs);
	if ( shape_ != rhs.shape() )
		throw std::invalid_argument("subtract_ requires tensors with the same shape.");
	if ( !is_contiguous() || !rhs.is_contiguous() )
		throw std::invalid_argument("subtract_ requires contiguous tensors.");

	if ( device_ == Device::CUDA )
		cuda::sub_arrays(device_data(), rhs.device_data(), device_data(), total_size_);
	else
		for ( int i = 0; i < total_size_; ++i )
			data()[i] -= rhs.data()[i];
}

} // namespace core
} // namespace tinytensor
