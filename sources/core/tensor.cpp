#include "cvmml/core/tensor.hpp"
#include "cvmml/core/cuda_kernels.cuh"
#include <atomic>
#include <algorithm>

namespace cvmml {
	namespace core {

		static std::atomic<uint64_t> tensor_global_id{0};

		static std::vector<int> make_contiguous_strides( const std::vector<int>& shape )
		{
			std::vector<int> strides(shape.size(), 1);
			if ( shape.empty() )
				return strides;
			int current_stride = 1;
			for ( int i = static_cast<int>(shape.size()) - 1; i >= 0; --i )
			{
				strides[i] = current_stride;
				current_stride *= shape[i];
			}
			return strides;
		}

		static int64_t product_of( const std::vector<int>& values )
		{
			int64_t product = 1;
			for ( int value : values )
				product *= value;
			return product;
		}

		static std::vector<int> unravel_index( int64_t linear, const std::vector<int>& shape )
		{
			std::vector<int> out(shape.size(), 0);
			for ( int i = static_cast<int>(shape.size()) - 1; i >= 0; --i )
			{
				out[i] = static_cast<int>(linear % shape[i]);
				linear /= shape[i];
			}
			return out;
		}

		static int64_t ravel_index( const std::vector<int>& indices, const std::vector<int>& shape )
		{
			if ( shape.empty() )
				return 0;
			int64_t linear = 0;
			for ( size_t i = 0; i < shape.size(); ++i )
				linear = linear * shape[i] + indices[i];
			return linear;
		}

		static std::vector<int> broadcast_batch_shape( const std::vector<int>& a, const std::vector<int>& b )
		{
			size_t out_rank = std::max(a.size(), b.size());
			std::vector<int> out(out_rank, 1);

			for ( size_t i = 0; i < out_rank; ++i )
			{
				int a_dim = (i < out_rank - a.size()) ? 1 : a[i - (out_rank - a.size())];
				int b_dim = (i < out_rank - b.size()) ? 1 : b[i - (out_rank - b.size())];
				if ( a_dim != b_dim && a_dim != 1 && b_dim != 1 )
					throw std::invalid_argument("Batch dimensions are not broadcast-compatible for matmult.");
				out[i] = std::max(a_dim, b_dim);
			}
			return out;
		}

		static int64_t map_broadcast_batch_index( const std::vector<int>& out_batch_index, const std::vector<int>& out_batch_shape, const std::vector<int>& operand_batch_shape )
		{
			if ( operand_batch_shape.empty() )
				return 0;

			std::vector<int> operand_index(operand_batch_shape.size(), 0);
			size_t shift = out_batch_shape.size() - operand_batch_shape.size();
			for ( size_t i = 0; i < operand_batch_shape.size(); ++i )
			{
				int dim = operand_batch_shape[i];
				int out_idx = out_batch_index[i + shift];
				operand_index[i] = (dim == 1) ? 0 : out_idx;
			}
			return ravel_index(operand_index, operand_batch_shape);
		}

		Tensor::Tensor( const std::vector<int>& shape ) : shape_(shape), total_size_(1), id_(++tensor_global_id)
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
			for ( int i = shape_.size() - 1; i >= 0; --i )
			{
				strides_[i] = current_stride;
				current_stride *= shape_[i];
			}
		}

		const std::vector<int>& Tensor::shape() const
		{
			return shape_;
		}

		const std::vector<int>& Tensor::strides() const
		{
			return strides_;
		}

		bool Tensor::is_view() const
		{
			return is_view_;
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

		Tensor Tensor::view( const std::vector<int>& new_shape ) const
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
			out.strides_ = make_contiguous_strides(resolved_shape);
			out.is_view_ = true;
			return out;
		}

		float* Tensor::data() const
		{
			if ( !data_ )
				return nullptr;
			return data_.get() + offset_;
		}

		int Tensor::size() const
		{
			return total_size_;
		}

		uint64_t Tensor::id() const
		{
			return id_;
		}

		float& Tensor::operator()( const std::vector<int>& indices )
		{
			if ( indices.size() != shape_.size() )
				throw std::invalid_argument("Number of indices must match the number of dimensions.");
			int flat_index = offset_;
			for ( size_t i = 0; i < indices.size(); i++ )
			{
				if ( indices[i] < 0 or indices[i] >= shape_[i] )
					throw std::out_of_range("Index out of shape bounds.");
				flat_index += indices[i] * strides_[i];
			}
			return data_[flat_index];
		}

		float Tensor::operator()( const std::vector<int>& indices ) const
		{
			if ( indices.size() != shape_.size() )
				throw std::invalid_argument("Number of indices must match the number of dimensions.");
			int flat_index = offset_;
			for ( size_t i = 0; i < indices.size(); i++ )
			{
				if ( indices[i] < 0 or indices[i] >= shape_[i] )
					throw std::out_of_range("Index out of shape bounds.");
				flat_index += indices[i] * strides_[i];
			}
			return data_[flat_index];
		}

		Tensor Tensor::zeros( const std::vector<int>& shape )
		{
			return Tensor(shape);
		}

		Tensor Tensor::ones( const std::vector<int>& shape )
		{
			Tensor t(shape);
			for ( int i = 0; i < t.size(); i++ )
				t.data()[i] = 1.0f;
			return t;
		}

		Tensor Tensor::randn( const std::vector<int>& shape, float mean, float std )
		{
			Tensor t(shape);
			std::random_device rd;
			std::mt19937 gen(rd());
			std::normal_distribution<float> d(mean, std);
			for ( int i = 0; i < t.size(); i++ )
				t.data()[i] = d(gen);
			return t;
		}

		void check_same_shape( const Tensor& a, const Tensor& b )
		{
			if ( a.shape() != b.shape() )
				throw std::invalid_argument("Tensors must have the same shape for this operation.");
		}

		void check_same_device( const Tensor& a, const Tensor& b )
		{
			if ( a.device() != b.device() )
				throw std::invalid_argument("Tensors must be on the same device for this operation.");
		}

		Tensor Tensor::operator+( const Tensor& rhs ) const
		{
			check_same_shape(*this, rhs);
			check_same_device(*this, rhs);
			Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
			Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

			Tensor result( this->shape_ );
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::add_arrays(lhs_cont.device_data(), rhs_cont.device_data(), result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = lhs_cont.data()[i] + rhs_cont.data()[i];

			result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);

			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				Tensor parentB = rhs;
				result.parents_ = {parentA, parentB};
				result.backward_fn_ = [parentA, parentB, result]() mutable
				{
					if ( parentA.requires_grad_ ) {
						if (parentA.device() == Device::CUDA)
							cuda::add_arrays(parentA.device_grad(), result.device_grad(), parentA.device_grad(), parentA.size());
						else
							for ( int i = 0; i < parentA.size(); i++ )
								parentA.grad()[i] += result.grad()[i];
					}
					if ( parentB.requires_grad_ ) {
						if (parentB.device() == Device::CUDA)
							cuda::add_arrays(parentB.device_grad(), result.device_grad(), parentB.device_grad(), parentB.size());
						else
							for ( int i = 0; i < parentB.size(); i++ )
								parentB.grad()[i] += result.grad()[i];
					}
				};
			}
			return result;
		}

		Tensor Tensor::operator-( const Tensor& rhs ) const
		{
			check_same_shape(*this, rhs);
			check_same_device(*this, rhs);
			Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
			Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

			Tensor result( this->shape_ );
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::sub_arrays(lhs_cont.device_data(), rhs_cont.device_data(), result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = lhs_cont.data()[i] - rhs_cont.data()[i];

			result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);

			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				Tensor parentB = rhs;
				result.parents_ = {parentA, parentB};
				result.backward_fn_ = [parentA, parentB, result]() mutable
				{
					if ( parentA.requires_grad_ ) {
						if (parentA.device() == Device::CUDA)
							cuda::add_arrays(parentA.device_grad(), result.device_grad(), parentA.device_grad(), parentA.size());
						else
							for ( int i = 0; i < parentA.size(); i++ )
								parentA.grad()[i] += result.grad()[i];
					}
					if ( parentB.requires_grad_ ) {
						if (parentB.device() == Device::CUDA)
							cuda::sub_arrays(parentB.device_grad(), result.device_grad(), parentB.device_grad(), parentB.size());
						else
							for ( int i = 0; i < parentB.size(); i++ )
								parentB.grad()[i] -= result.grad()[i];
					}
				};
			}
			return result;
		}

		Tensor Tensor::operator*( const Tensor& rhs ) const
		{
			check_same_shape(*this, rhs);
			check_same_device(*this, rhs);
			Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
			Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

			Tensor result( this->shape_ );
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::mul_arrays(lhs_cont.device_data(), rhs_cont.device_data(), result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = lhs_cont.data()[i] * rhs_cont.data()[i];

			result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);

			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				Tensor parentB = rhs;
				result.parents_ = {parentA, parentB};
				result.backward_fn_ = [parentA, parentB, result]() mutable
				{
					if ( parentA.requires_grad_ ) {
						if (parentA.device() == Device::CUDA)
							cuda::add_mul_arrays(parentA.device_grad(), result.device_grad(), parentB.device_data(), parentA.size());
						else
							for ( int i = 0; i < parentA.size(); i++ )
								parentA.grad()[i] += result.grad()[i] * parentB.data()[i];
					}
					if ( parentB.requires_grad_ ) {
						if (parentB.device() == Device::CUDA)
							cuda::add_mul_arrays(parentB.device_grad(), result.device_grad(), parentA.device_data(), parentB.size());
						else
							for ( int i = 0; i < parentB.size(); i++ )
								parentB.grad()[i] += result.grad()[i] * parentA.data()[i];
					}
				};
			}
			return result;
		}

		Tensor Tensor::operator/( const Tensor& rhs ) const
		{
			check_same_shape(*this, rhs);
			check_same_device(*this, rhs);
			Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
			Tensor rhs_cont = rhs.is_contiguous() ? rhs : rhs.contiguous();

			Tensor result( this->shape_ );
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::div_arrays(lhs_cont.device_data(), rhs_cont.device_data(), result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = lhs_cont.data()[i] / rhs_cont.data()[i];

			result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);

			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				Tensor parentB = rhs;
				result.parents_ = {parentA, parentB};
				result.backward_fn_ = [parentA, parentB, result]() mutable
				{
					if ( parentA.requires_grad_ ) {
						if (parentA.device() == Device::CUDA)
							cuda::add_div_arrays(parentA.device_grad(), result.device_grad(), parentB.device_data(), parentA.size());
						else
							for ( int i = 0; i < parentA.size(); i++ )
								parentA.grad()[i] += result.grad()[i] / parentB.data()[i];
					}
					if ( parentB.requires_grad_ ) {
						if (parentB.device() == Device::CUDA)
							cuda::sub_mul_div_sqr_arrays(parentB.device_grad(), result.device_grad(), parentA.device_data(), parentB.device_data(), parentB.size());
						else
							for ( int i = 0; i < parentB.size(); i++ )
								parentB.grad()[i] += result.grad()[i] * (-parentA.data()[i] / (parentB.data()[i] * parentB.data()[i]));
					}
				};
			}
			return result;
		}

		Tensor Tensor::operator+( float scalar ) const
		{
			Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
			Tensor result(this->shape_);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::add_scalar(lhs_cont.device_data(), scalar, result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = lhs_cont.data()[i] + scalar;
			result.set_requires_grad(this->requires_grad_);
			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				result.parents_ = {parentA};
				result.backward_fn_ = [parentA, result]() mutable
				{
					if ( parentA.requires_grad_ ) {
						if (parentA.device() == Device::CUDA)
							cuda::add_arrays(parentA.device_grad(), result.device_grad(), parentA.device_grad(), parentA.size());
						else
							for ( int i = 0; i < parentA.size(); i++ )
								parentA.grad()[i] += result.grad()[i];
					}
				};
			}
			return result;
		}

		Tensor Tensor::operator-( float scalar ) const
		{
			Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
			Tensor result(this->shape_);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::sub_scalar(lhs_cont.device_data(), scalar, result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = lhs_cont.data()[i] - scalar;
			result.set_requires_grad(this->requires_grad_);
			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				result.parents_ = {parentA};
				result.backward_fn_ = [parentA, result]() mutable
				{
					if ( parentA.requires_grad_ ) {
						if (parentA.device() == Device::CUDA)
							cuda::add_arrays(parentA.device_grad(), result.device_grad(), parentA.device_grad(), parentA.size());
						else
							for ( int i = 0; i < parentA.size(); i++ )
								parentA.grad()[i] += result.grad()[i];
					}
				};
			}
			return result;
		}

		Tensor Tensor::operator*( float scalar ) const
		{
			Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
			Tensor result(this->shape_);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::mul_scalar(lhs_cont.device_data(), scalar, result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = lhs_cont.data()[i] * scalar;
			result.set_requires_grad(this->requires_grad_);
			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				result.parents_ = {parentA};
				result.backward_fn_ = [parentA, result, scalar]() mutable
				{
					if ( parentA.requires_grad_ ) {
						if (parentA.device() == Device::CUDA)
							cuda::add_mul_scalar_arrays(parentA.device_grad(), result.device_grad(), scalar, parentA.size());
						else
							for ( int i = 0; i < parentA.size(); i++ )
								parentA.grad()[i] += result.grad()[i] * scalar;
					}
				};
			}
			return result;
		}

		Tensor Tensor::operator/( float scalar ) const
		{
			Tensor lhs_cont = this->is_contiguous() ? *this : this->contiguous();
			Tensor result(this->shape_);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::div_scalar(lhs_cont.device_data(), scalar, result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = lhs_cont.data()[i] / scalar;
			result.set_requires_grad(this->requires_grad_);
			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				result.parents_ = {parentA};
				result.backward_fn_ = [parentA, result, scalar]() mutable
				{
					if ( parentA.requires_grad_ ) {
						if (parentA.device() == Device::CUDA)
							cuda::add_div_scalar_arrays(parentA.device_grad(), result.device_grad(), scalar, parentA.size());
						else
							for ( int i = 0; i < parentA.size(); i++ )
								parentA.grad()[i] += result.grad()[i] / scalar;
					}
				};
			}
			return result;
		}

		Tensor Tensor::matmult( const Tensor& rhs ) const
		{
			if ( this->shape_.size() < 2 || rhs.shape().size() < 2 )
				throw std::invalid_argument("matmult requires tensors with at least 2 dimensions.");
			check_same_device(*this, rhs);

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
			std::vector<int> out_batch_shape = broadcast_batch_shape(lhs_batch_shape, rhs_batch_shape);

			std::vector<int> out_shape = out_batch_shape;
			out_shape.push_back(M);
			out_shape.push_back(N);

			int64_t batch_count = out_batch_shape.empty() ? 1 : product_of(out_batch_shape);
			int64_t lhs_matrix_size = static_cast<int64_t>(M) * K;
			int64_t rhs_matrix_size = static_cast<int64_t>(K) * N;
			int64_t out_matrix_size = static_cast<int64_t>(M) * N;

			Tensor result(out_shape);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				for ( int64_t b = 0; b < batch_count; ++b )
				{
					std::vector<int> out_batch_idx = unravel_index(b, out_batch_shape);
					int64_t lhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
					int64_t rhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);

					const float* A_ptr = lhs_cont.device_data() + lhs_batch_linear * lhs_matrix_size;
					const float* B_ptr = rhs_cont.device_data() + rhs_batch_linear * rhs_matrix_size;
					float* C_ptr = result.device_data() + b * out_matrix_size;
					cuda::matmul(A_ptr, B_ptr, C_ptr, M, K, N);
				}
			}
			else
			{
				for ( int64_t b = 0; b < batch_count; ++b )
				{
					std::vector<int> out_batch_idx = unravel_index(b, out_batch_shape);
					int64_t lhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
					int64_t rhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);

					const float* A_ptr = lhs_cont.data() + lhs_batch_linear * lhs_matrix_size;
					const float* B_ptr = rhs_cont.data() + rhs_batch_linear * rhs_matrix_size;
					float* C_ptr = result.data() + b * out_matrix_size;

					for ( int i = 0; i < M; i++ )
					{
						for ( int j = 0; j < N; j++ )
						{
							float sum = 0.0f;
							for ( int k = 0; k < K; k++ )
								sum += A_ptr[i * K + k] * B_ptr[k * N + j];
							C_ptr[i * N + j] = sum;
						}
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
					int64_t batch_count_local = out_batch_shape.empty() ? 1 : product_of(out_batch_shape);
					int64_t lhs_matrix_size_local = static_cast<int64_t>(M) * K;
					int64_t rhs_matrix_size_local = static_cast<int64_t>(K) * N;
					int64_t out_matrix_size_local = static_cast<int64_t>(M) * N;

					if ( parentA.requires_grad_ )
					{
						if (parentA.device() == Device::CUDA)
						{
							for ( int64_t b = 0; b < batch_count_local; ++b )
							{
								std::vector<int> out_batch_idx = unravel_index(b, out_batch_shape);
								int64_t rhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);
								int64_t lhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);

								cuda::matmult_backward_A(
									result.device_grad() + b * out_matrix_size_local,
									parentB.device_data() + rhs_batch_linear * rhs_matrix_size_local,
									parentA.device_grad() + lhs_batch_linear * lhs_matrix_size_local,
									M, K, N
								);
							}
						}
						else
						{
							for ( int64_t b = 0; b < batch_count_local; ++b )
							{
								std::vector<int> out_batch_idx = unravel_index(b, out_batch_shape);
								int64_t rhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);
								int64_t lhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);

								const float* gradC = result.grad() + b * out_matrix_size_local;
								const float* B_ptr = parentB.data() + rhs_batch_linear * rhs_matrix_size_local;
								float* gradA = parentA.grad() + lhs_batch_linear * lhs_matrix_size_local;

								for ( int i = 0; i < M; i++ )
								{
									for ( int j = 0; j < N; j++ )
									{
										float dC = gradC[i * N + j];
										for ( int k = 0; k < K; k++ )
											gradA[i * K + k] += dC * B_ptr[k * N + j];
									}
								}
							}
						}
					}

					if ( parentB.requires_grad_ )
					{
						if (parentB.device() == Device::CUDA)
						{
							for ( int64_t b = 0; b < batch_count_local; ++b )
							{
								std::vector<int> out_batch_idx = unravel_index(b, out_batch_shape);
								int64_t lhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
								int64_t rhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);

								cuda::matmult_backward_B(
									parentA.device_data() + lhs_batch_linear * lhs_matrix_size_local,
									result.device_grad() + b * out_matrix_size_local,
									parentB.device_grad() + rhs_batch_linear * rhs_matrix_size_local,
									M, K, N
								);
							}
						}
						else
						{
							for ( int64_t b = 0; b < batch_count_local; ++b )
							{
								std::vector<int> out_batch_idx = unravel_index(b, out_batch_shape);
								int64_t lhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, lhs_batch_shape);
								int64_t rhs_batch_linear = map_broadcast_batch_index(out_batch_idx, out_batch_shape, rhs_batch_shape);

								const float* A_ptr = parentA.data() + lhs_batch_linear * lhs_matrix_size_local;
								const float* gradC = result.grad() + b * out_matrix_size_local;
								float* gradB = parentB.grad() + rhs_batch_linear * rhs_matrix_size_local;

								for ( int i = 0; i < M; i++ )
								{
									for ( int j = 0; j < N; j++ )
									{
										float dC = gradC[i * N + j];
										for ( int k = 0; k < K; k++ )
											gradB[k * N + j] += dC * A_ptr[i * K + k];
									}
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
			if ( shape_.size() != 2 )
				throw std::invalid_argument("Transpose only supports 2D tensors for now.");

			int rank = static_cast<int>(shape_.size());
			if ( dim0 < 0 )
				dim0 += rank;
			if ( dim1 < 0 )
				dim1 += rank;
			if ( dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank || dim0 == dim1 )
				throw std::invalid_argument("Invalid transpose dimensions.");
			if ( !((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0)) )
				throw std::invalid_argument("For 2D tensors, transpose only supports dim pairs (0,1), (1,0), (-2,-1), (-1,-2).");

			int M = shape_[0];
			int N = shape_[1];
			Tensor tr = *this;
			std::swap(tr.shape_[dim0], tr.shape_[dim1]);
			std::swap(tr.strides_[dim0], tr.strides_[dim1]);
			tr.is_view_ = true;

			if ( this->device_ == Device::CUDA )
			{
				Tensor tr_contig({N, M});
				tr_contig = tr_contig.to_cuda();
				cuda::transpose_matrix(this->device_data(), tr_contig.device_data(), M, N);
				tr = tr_contig;
			}
					
			tr.set_requires_grad(this->requires_grad_);
			
			if ( tr.requires_grad_ )
			{
				Tensor parent = *this;
				tr.parents_ = {parent};
				tr.backward_fn_ = [parent, tr, M, N]() mutable
				{
					if ( parent.requires_grad_ )
					{
						if ( parent.device() == Device::CUDA )
							cuda::add_transpose_matrix(tr.device_grad(), parent.device_grad(), M, N);
						else
						{
							for ( int i = 0; i < M; i++ )
								for ( int j = 0; j < N; j++ )
									parent.grad()[i * N + j] += tr.grad()[j * M + i];
						}
					}
				};
			}
			return tr;
		}

		bool Tensor::requires_grad() const
		{
			 return requires_grad_;
		}

		void Tensor::set_requires_grad( bool val )
		{
			requires_grad_ = val;
			if ( val && !grad_ )
				grad_ = std::shared_ptr<float[]>(new float[total_size_]());
			if ( val && device_ == Device::CUDA && !device_grad_ )
			{
				device_grad_ = std::shared_ptr<float[]>(
					cuda::allocate_memory(total_size_),
					[](float* ptr) {cuda::free_memory(ptr);}
				);
				cuda::set_memory(device_grad_.get(), 0.0f, total_size_);
			}
		}

		float* Tensor::grad() const
		{
			if ( grad_ )
				return grad_.get();
			return nullptr;
		}

		void Tensor::zero_grad()
		{
			if ( grad_ )
				for ( int i = 0; i < total_size_; i++ )
					grad_[i] = 0.0f;
			if ( device_grad_ )
				cuda::set_memory(device_grad_.get(), 0.0f, total_size_);
		}

		void Tensor::backward()
		{
			if ( !requires_grad_ )
				return ;
			if ( !grad_ || (device_ == Device::CUDA && !device_grad_) )
				set_requires_grad( true );
			if ( device_ == Device::CUDA )
				cuda::fill_ones(device_grad_.get(), total_size_);
			else {
				for ( int i = 0; i < total_size_; i++ )
					grad_[i] = 1.0f;
			}
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
			for ( auto it = topo.rbegin(); it != topo.rend(); it++ )
				if ( it->backward_fn_ )
					it->backward_fn_();
		}

		float* Tensor::device_grad() const
		{
			if ( device_grad_ )
				return device_grad_.get();
			return nullptr;
		}

	} // namespace core
} // namespace cvmml
