#include "tensor.hpp"
#include <stdexcept>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)
	do {
		cudaError_t err = call;
		if ( err != cudaSuccess )
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
	} while (0)
#endif

namespace cvmml {
	namespace core {

		Device Tensor::device() const
		{
			return device_;
		}

		float* Tensor::device_data() const
		{
			return device_data_.get();
		}

		Tensor Tensor::to_cuda() const
		{
			if ( device_ == Device::CUDA )
				return *this;
			Tensor result = *this;
			result.device_ = Device::CUDA;
			float* d_ptr = nullptr;
			size_t bytes = total_size_ * sizeof(float);
			CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
			CHECK_CUDA(cudaMemcpy(d_ptr, this->data_.get(), bytes, cudaMemcpyHostToDevice));
			result.device_data_ = std::shared_ptr<float>(d_ptr, [](float* ptr)
			{
				cudaFree(ptr);
			});
			result.data_.reset();
			return result;
		}

		Tensor Tensor::to_cpu() const
		{
			if ( device_ == Device::CPU )
				return *this;
			Tensor result = *this;
			result.device_ = Device::CPU;
			result.data_ = std::shared_ptr<float[]>(new float[total_size_]());
			size_t bytes = total_size_ * sizeof(float);
			CHECK_CUDA(cudaMemcpy(result.data_.get(), this->device_data_.get(), bytes, cudaMemcpyDeviceToHost));
			result.device_data_.reset();
			return result;
		}

	} // namespace core
} // namespace cvmml