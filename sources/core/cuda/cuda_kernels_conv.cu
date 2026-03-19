#include "core/cuda_kernels.cuh"
#include "core/cuda_kernels_common.cuh"

namespace cvmml {
namespace core {
namespace cuda {

__global__ void im2col_kernel( const float* data_im, float* data_col, int channels, int height, int width, int kH, int kW, int stride, int pad, int height_out, int width_out)
{
	int total = channels * kH * kW * height_out * width_out;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= total )
		return;

	int col_channels = channels * kH * kW;
	int hw_out = idx % (height_out * width_out);
	int c_col  = idx / (height_out * width_out);

	int ow = hw_out % width_out;
	int oh = hw_out / width_out;

	int kw = c_col % kW;
	int kh = (c_col / kW) % kH;
	int c  = c_col / (kH * kW);

	int ih = oh * stride - pad + kh;
	int iw = ow * stride - pad + kw;

	float val = 0.0f;
	if ( ih >= 0 && ih < height && iw >= 0 && iw < width )
		val = data_im[c * height * width + ih * width + iw];

	data_col[c_col * (height_out * width_out) + hw_out] = val;
}

void im2col(const float* data_im, float* data_col, int batch, int channels, int height, int width, int kH, int kW, int stride, int pad, int height_out, int width_out)
{
	int col_channels = channels * kH * kW;
	int per_image = col_channels * height_out * width_out;

	dim3 blockSize, gridSize;
	get_grid_1d(per_image, blockSize, gridSize);

	int im_size = channels * height * width;

	for ( int b = 0; b < batch; ++b )
	{
		im2col_kernel<<<gridSize, blockSize>>>(
			data_im + b * im_size,
			data_col + b * per_image,
			channels, height, width,
			kH, kW, stride, pad,
			height_out, width_out);
		CHECK_CUDA_LAUNCH();
	}
}

} // namespace cuda
} // namespace core
} // namespace cvmml
