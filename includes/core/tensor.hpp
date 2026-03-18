#ifndef CVMML_CORE_TENSOR_HPP
#define CVMML_CORE_TENSOR_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <random>
#include <functional>
#include <set>
#include <string>
#include <atomic>
#include <algorithm>
#include <cmath>

enum class Device
{
	CPU,
	CUDA
};

namespace cvmml {
namespace core {

class Tensor {
	private:
		std::vector<int> shape_;
		std::vector<int> strides_;
		int offset_ = 0;
		bool is_view_ = false;
		std::shared_ptr<float[]> data_;

		Device device_ = Device::CPU;
		std::shared_ptr<float[]> device_data_;

		int total_size_;
		uint64_t id_;
		void compute_strides();

		bool requires_grad_ = false;
		std::shared_ptr<float[]> grad_;
		std::shared_ptr<float[]> device_grad_;
		std::vector<Tensor> parents_;
		std::function<void()> backward_fn_;

	public:
		Tensor( const std::vector<int>& shape );

		const std::vector<int>& shape() const;
		const std::vector<int>& strides() const;
		bool is_view() const;
		bool is_contiguous() const;
		Tensor contiguous() const;
		Tensor view( const std::vector<int>& new_shape ) const;
		float* data() const;
		int size() const;
		uint64_t id() const;

		Device device() const;
		float* device_data() const;
		Tensor to_cuda() const;
		Tensor to_cpu() const;

		float& operator()( const std::vector<int>& indices );
		float operator()( const std::vector<int>& indices ) const;

		static Tensor zeros( const std::vector<int>& shape );
		static Tensor ones( const std::vector<int>& shape );
		static Tensor randn( const std::vector<int>& shape, float mean, float std );

		Tensor operator+( const Tensor& rhs ) const;
		Tensor operator-( const Tensor& rhs ) const;
		Tensor operator*( const Tensor& rhs ) const;
		Tensor operator/( const Tensor& rhs ) const;

		Tensor operator+( float scalar ) const;
		Tensor operator-( float scalar ) const;
		Tensor operator*( float scalar ) const;
		Tensor operator/( float scalar ) const;

		Tensor exp() const;
		Tensor log() const;
		Tensor sum( const std::vector<int>& axes = {}, bool keepdim = false ) const;
		Tensor max( const std::vector<int>& axes = {}, bool keepdim = false ) const;
		Tensor min( const std::vector<int>& axes = {}, bool keepdim = false ) const;

		Tensor matmult( const Tensor& rhs ) const;
		Tensor transpose(int dim0 = -2, int dim1 = -1) const;

		bool requires_grad() const;
		void set_requires_grad( bool val );
		float* grad() const;
		float* device_grad() const;
		void zero_grad();
		void backward();
};

} // namespace core
} // namespace cvmml

#endif // CVMML_CORE_TENSOR_HPP