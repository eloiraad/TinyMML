#include "core/cuda_kernels.cuh"
#include "core/tensor.hpp"

#include <functional>
#include <set>
#include <vector>

namespace cvmml {
namespace core {

bool Tensor::requires_grad() const
{
	return requires_grad_;
}

void Tensor::set_requires_grad(bool val)
{
	requires_grad_ = val;
	if ( val && !grad_ )
		grad_ = std::shared_ptr<float[]>(new float[total_size_]());
	if ( val && device_ == Device::CUDA && !device_grad_ )
	{
		device_grad_ = std::shared_ptr<float[]>(cuda::allocate_memory(total_size_), [](float* ptr) { cuda::free_memory(ptr); });
		cuda::set_memory(device_grad_.get(), 0.0f, total_size_);
	}
}

float* Tensor::grad() const
{
	if ( device_ == Device::CUDA && device_grad_ && grad_ )
		cuda::copy_to_host(grad_.get(), device_grad_.get(), total_size_);
	if ( grad_ )
		return grad_.get();
	return nullptr;
}

float* Tensor::device_grad() const
{
	if ( device_grad_ )
		return device_grad_.get();
	return nullptr;
}

void Tensor::zero_grad()
{
	if ( grad_ )
		for ( int i = 0; i < total_size_; ++i )
			grad_[i] = 0.0f;
	if ( device_grad_ )
		cuda::set_memory(device_grad_.get(), 0.0f, total_size_);
}

void Tensor::backward()
{
	if ( !requires_grad_ )
		return;
	if ( total_size_ != 1 )
		throw std::runtime_error("backward() can only be called on a scalar (size=1) tensor. Use .sum() first.");
	if ( !grad_ || (device_ == Device::CUDA && !device_grad_) )
		set_requires_grad(true);

	if ( device_ == Device::CUDA )
		cuda::fill_ones(device_grad_.get(), total_size_);
	else
		for ( int i = 0; i < total_size_; ++i )
			grad_[i] = 1.0f;

	std::vector<Tensor> topo;
	std::set<uint64_t> visited;
	std::function<void(const Tensor&)> build_topo = [&](const Tensor& t)
	{
		if ( visited.find(t.id()) == visited.end() )
		{
			visited.insert(t.id());
			for ( const Tensor& parent : t.parents_ )
				build_topo(parent);
			topo.push_back(t);
		}
	};

	build_topo(*this);
	for ( auto it = topo.rbegin(); it != topo.rend(); ++it )
		if ( it->backward_fn_ )
			it->backward_fn_();
}

} // namespace core
} // namespace cvmml
