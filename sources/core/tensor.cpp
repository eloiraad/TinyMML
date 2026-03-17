#include "cvmml/core/tensor.hpp"
#include "cvmml/core/cuda_kernels.cuh"
#include <atomic>

namespace cvmml {
	namespace core {

		static std::atomic<uint64_t> tensor_global_id{0};

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

		float* Tensor::data() const
		{
			return data_.get();
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
			int flat_index = 0;
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
			int flat_index = 0;
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

			Tensor result( this->shape_ );
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::add_arrays(this->device_data(), rhs.device_data(), result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = this->data()[i] + rhs.data()[i];

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

			Tensor result( this->shape_ );
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::sub_arrays(this->device_data(), rhs.device_data(), result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = this->data()[i] - rhs.data()[i];

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

			Tensor result( this->shape_ );
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::mul_arrays(this->device_data(), rhs.device_data(), result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = this->data()[i] * rhs.data()[i];

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

			Tensor result( this->shape_ );
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::div_arrays(this->device_data(), rhs.device_data(), result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = this->data()[i] / rhs.data()[i];

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
			Tensor result(this->shape_);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::add_scalar(this->device_data(), scalar, result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = this->data()[i] + scalar;
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
			Tensor result(this->shape_);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::sub_scalar(this->device_data(), scalar, result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = this->data()[i] - scalar;
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
			Tensor result(this->shape_);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::mul_scalar(this->device_data(), scalar, result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = this->data()[i] * scalar;
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
			Tensor result(this->shape_);
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::div_scalar(this->device_data(), scalar, result.device_data(), total_size_);
			}
			else
				for ( int i = 0; i < this->total_size_; i++ )
					result.data()[i] = this->data()[i] / scalar;
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
			if ( this->shape_.size() != 2 or rhs.shape().size() != 2 )
				throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
			check_same_device(*this, rhs);

			int M = this->shape_[0];
			int K = this->shape_[1];
			int K2 = rhs.shape()[0];
			int N = rhs.shape()[1];
			if ( K != K2 )
				throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
			
			Tensor result({M, N});
			if ( this->device_ == Device::CUDA )
			{
				result = result.to_cuda();
				cuda::matmul(this->device_data(), rhs.device_data(), result.device_data(), M, K, N);
			}
			else
			{
				for ( int i = 0; i < M; i++ )
				{
					for ( int j = 0; j < N; j++ )
					{
						float sum = 0.0f;
						for ( int k = 0; k < K; k++ )
							sum += (*this)({i, k}) * rhs({k, j});
						result({i, j}) = sum;
					}
				}
			}

			result.set_requires_grad(this->requires_grad_ || rhs.requires_grad_);

			if ( result.requires_grad_ )
			{
				Tensor parentA = *this;
				Tensor parentB = rhs;
				result.parents_ = {parentA, parentB};
				result.backward_fn_ = [parentA, parentB, result, M, K, N]() mutable
				{
					if ( parentA.requires_grad_ )
					{
						if (parentA.device() == Device::CUDA)
							cuda::matmult_backward_A(result.device_grad(), parentB.device_data(), parentA.device_grad(), M, K, N);
						else
						{
							for ( int i = 0; i < M; i++ )
							{
								for ( int j = 0; j < N; j++ )
								{
									float dC = result.grad()[i * N + j];
									for ( int k = 0; k < K; k++ )
										parentA.grad()[i * K + k] += dC * parentB.data()[k * N + j];
								}
							}
						}
					}
					if ( parentB.requires_grad_ )
					{
						if (parentB.device() == Device::CUDA)
							cuda::matmult_backward_B(parentA.device_data(), result.device_grad(), parentB.device_grad(), M, K, N);
						else
						{
							for ( int i = 0; i < M; i++ )
							{
								for ( int j = 0; j < N; j++ )
								{
									float dC = result.grad()[i * N + j];
									for ( int k = 0; k < K; k++ )
										parentB.grad()[k * N + j] += dC * parentA.data()[i * K + k];
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
			Tensor tr({N, M});
			
			if ( this->device_ == Device::CUDA )
			{
				tr = tr.to_cuda();
				cuda::transpose_matrix(this->device_data(), tr.device_data(), M, N);
			}
			else
			{
				for ( int i = 0; i < M; i++ )
					for ( int j = 0; j < N; j++ )
						tr({j, i}) = (*this)({i, j});
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
