#include "core/cuda_kernels.cuh"
#include "core/tensor.hpp"
#include "core/tensor_detail.hpp"

namespace cvmml {
namespace core {

Tensor Tensor::im2col(int kH, int kW, int stride, int pad) const
{
	// What: Extract convolution patches from [B, C, H, W] into a columns matrix
	//       of shape [B, C*kH*kW, H_out*W_out].
	// Why:  Reduces convolution to a GEMM, enabling reuse of the optimised matmult path.
	if ( shape_.size() != 4 )
		throw std::invalid_argument("im2col requires a 4‑D tensor [B, C, H, W].");

	int B  = shape_[0];
	int C  = shape_[1];
	int H  = shape_[2];
	int W  = shape_[3];
	int H_out = (H + 2 * pad - kH) / stride + 1;
	int W_out = (W + 2 * pad - kW) / stride + 1;

	if ( H_out <= 0 || W_out <= 0 )
		throw std::invalid_argument("im2col: kernel/stride/pad combination yields non‑positive output size.");

	int col_channels = C * kH * kW;
	Tensor result({B, col_channels, H_out * W_out});

	Tensor src = this->is_contiguous() ? *this : this->contiguous();

	if ( device_ == Device::CUDA )
	{
		result = result.to_cuda();
		cuda::im2col(src.device_data(), result.device_data(), B, C, H, W, kH, kW, stride, pad, H_out, W_out);
	}
	else
	{
		const float* im = src.data();
		float* col = result.data();

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for ( int b = 0; b < B; ++b )
		{
			const float* im_b = im + b * C * H * W;
			float* col_b = col + b * col_channels * H_out * W_out;

			for ( int c = 0; c < C; ++c )
			{
				for ( int kh = 0; kh < kH; ++kh )
				{
					for ( int kw = 0; kw < kW; ++kw )
					{
						int col_row = c * kH * kW + kh * kW + kw;
						for ( int oh = 0; oh < H_out; ++oh )
						{
							for ( int ow = 0; ow < W_out; ++ow )
							{
								int ih = oh * stride - pad + kh;
								int iw = ow * stride - pad + kw;
								int col_idx = col_row * (H_out * W_out) + oh * W_out + ow;
								if ( ih >= 0 && ih < H && iw >= 0 && iw < W )
									col_b[col_idx] = im_b[c * H * W + ih * W + iw];
								else
									col_b[col_idx] = 0.0f; // zero-padding
							}
						}
					}
				}
			}
		}
	}
	return result;
}

} // namespace core
} // namespace cvmml
