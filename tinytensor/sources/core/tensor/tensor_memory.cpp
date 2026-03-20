#include "cuda_kernels.cuh"
#include "tensor.hpp"
#include "tensor_detail.hpp"

#include <atomic>
#include <random>

namespace tinytensor {
namespace core {

static std::atomic<uint64_t> tensor_global_id{0};

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape), total_size_(1), id_(++tensor_global_id)
{
	if ( shape.empty() )
		total_size_ = 0;
	else
	{
		for ( int dim : shape )
		{
			if ( dim <= 0 )
				throw std::invalid_argument("Shape dimensions must be positive.");
			total_size_ *= dim;
		}
	}
	data_ = std::shared_ptr<float[]>(new float[total_size_]());
	compute_strides();
}

void Tensor::compute_strides()
{
	strides_.resize(shape_.size());
	if ( shape_.empty() )
		return;
	int current_stride = 1;
	for ( int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i )
	{
		strides_[i] = current_stride;
		current_stride *= shape_[i];
	}
}

const std::vector<int>& Tensor::shape() const { return shape_; }
const std::vector<int>& Tensor::strides() const { return strides_; }
bool Tensor::is_view() const { return is_view_; }
int Tensor::size() const { return total_size_; }
uint64_t Tensor::id() const { return id_; }

float* Tensor::data() const
{
	if ( !data_ )
		return nullptr;
	return data_.get() + offset_;
}

bool Tensor::is_contiguous() const
{
	if ( shape_.empty() )
		return true;
	int expected_stride = 1;
	for ( int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i )
	{
		if ( strides_[i] != expected_stride )
			return false;
		expected_stride *= shape_[i];
	}
	return true;
}

Tensor Tensor::contiguous() const
{
	if ( is_contiguous() && offset_ == 0 )
		return *this;

	if ( device_ == Device::CUDA )
	{
		Tensor out(shape_);
		out = out.to_cuda();
		cuda::pack_strided_to_contiguous(device_data_.get(), out.device_data_.get(), shape_.data(), strides_.data(), static_cast<int>(shape_.size()), offset_, total_size_);
		return out;
	}

	Tensor out(shape_);
	for ( int linear = 0; linear < total_size_; ++linear )
	{
		int rem = linear;
		int source_index = offset_;
		for ( int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d )
		{
			int idx = rem % shape_[d];
			rem /= shape_[d];
			source_index += idx * strides_[d];
		}
		out.data()[linear] = data_[source_index];
	}
	return out;
}

Tensor Tensor::view(const std::vector<int>& new_shape) const
{
	if ( !is_contiguous() )
		throw std::invalid_argument("view() requires a contiguous tensor. Call contiguous() first.");

	int infer_dim = -1;
	long long known_product = 1;
	for ( int i = 0; i < static_cast<int>(new_shape.size()); ++i )
	{
		if ( new_shape[i] == -1 )
		{
			if ( infer_dim != -1 )
				throw std::invalid_argument("Only one inferred dimension (-1) is allowed in view().");
			infer_dim = i;
		}
		else if ( new_shape[i] <= 0 )
			throw std::invalid_argument("view() dimensions must be positive or -1.");
		else
			known_product *= new_shape[i];
	}

	std::vector<int> resolved_shape = new_shape;
	if ( infer_dim != -1 )
	{
		if ( known_product == 0 || total_size_ % known_product != 0 )
			throw std::invalid_argument("view() inferred dimension is incompatible with tensor size.");
		resolved_shape[infer_dim] = total_size_ / static_cast<int>(known_product);
	}

	long long product = 1;
	for ( int dim : resolved_shape )
		product *= dim;
	if ( product != total_size_ )
		throw std::invalid_argument("view() shape is incompatible with tensor size.");

	Tensor out = *this;
	out.shape_ = resolved_shape;
	out.strides_ = detail::make_contiguous_strides(resolved_shape);
	out.is_view_ = true;
	out.grad_.reset();
	out.device_grad_.reset();
	out.parents_.clear();
	out.backward_fn_ = std::function<void(const Tensor&)>();
	return out;
}

float& Tensor::operator()(const std::vector<int>& indices)
{
	if ( indices.size() != shape_.size() )
		throw std::invalid_argument("Number of indices must match the number of dimensions.");
	int flat_index = offset_;
	for ( size_t i = 0; i < indices.size(); ++i )
	{
		if ( indices[i] < 0 || indices[i] >= shape_[i] )
			throw std::out_of_range("Index out of shape bounds.");
		flat_index += indices[i] * strides_[i];
	}
	return data_[flat_index];
}

float Tensor::operator()(const std::vector<int>& indices) const
{
	if ( indices.size() != shape_.size() )
		throw std::invalid_argument("Number of indices must match the number of dimensions.");
	int flat_index = offset_;
	for ( size_t i = 0; i < indices.size(); ++i )
	{
		if ( indices[i] < 0 || indices[i] >= shape_[i] )
			throw std::out_of_range("Index out of shape bounds.");
		flat_index += indices[i] * strides_[i];
	}
	return data_[flat_index];
}

Tensor Tensor::zeros(const std::vector<int>& shape)
{
	return Tensor(shape);
}

Tensor Tensor::ones(const std::vector<int>& shape)
{
	Tensor t(shape);
	for ( int i = 0; i < t.size(); ++i )
		t.data()[i] = 1.0f;
	return t;
}

Tensor Tensor::randn(const std::vector<int>& shape, float mean, float std)
{
	Tensor t(shape);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> d(mean, std);
	for ( int i = 0; i < t.size(); ++i )
		t.data()[i] = d(gen);
	return t;
}

Tensor Tensor::bernoulli(const std::vector<int>& shape, float p)
{
	if ( p < 0.0f || p > 1.0f )
		throw std::invalid_argument("bernoulli probability p must be in [0, 1].");
	Tensor t(shape);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution d(p);
	for ( int i = 0; i < t.size(); ++i )
		t.data()[i] = d(gen) ? 1.0f : 0.0f;
	return t;
}

Tensor Tensor::uniform(const std::vector<int>& shape)
{
	Tensor t(shape);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> d(0.0f, 1.0f);
	for ( int i = 0; i < t.size(); ++i )
		t.data()[i] = d(gen);
	return t;
}

} // namespace core
} // namespace tinytensor
