#include "cuda_kernels.cuh"
#include "tensor.hpp"
#include "tensor_detail.hpp"

#include <stdexcept>

namespace tinytensor {
namespace core {

Tensor Tensor::im2col(int kH, int kW, int stride, int pad) const
{
/**
 * @brief Extrait et aplanit les patchs d'une image pour convolution.
 *
 * @details
 * Permet d'implémenter la convolution 2D comme une simple multiplication matricielle, exploitant ainsi les optimisations de `matmult`.
 * Parcours la matrice spatiale par fenêtres glissantes et copie chaque patch sous forme de colonne. La boucle CPU sur les batchs est parallélisée.
 *
 * @param kH (int) Hauteur du noyau de convolution.
 * @param kW (int) Largeur du noyau de convolution.
 * @param stride (int) Décalage spatial entre chaque prise de patch.
 * @param pad (int) Remplissage de zéros symétrique ajouté tout autour de l'image (hauteur et largeur).
 * @return (Tensor) [B, C*kH*kW, H_out*W_out] Tenseur 3D contenant toutes les colonnes extraites pour chaque image.
 */
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
		if ( requires_grad_ )
			throw std::runtime_error("CUDA im2col backward is not implemented.");
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

	result.set_requires_grad(requires_grad_);
	if ( result.requires_grad_ )
	{
		result.parents_ = {*this};
		result.backward_fn_ = [B, C, H, W, kH, kW, stride, pad, H_out, W_out, col_channels](const Tensor& result)
		{
			const Tensor& parent = result.parents_[0];
			const float* grad_col = result.grad();
			float* grad_image = parent.grad();

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for ( int b = 0; b < B; ++b )
			{
				const float* grad_col_b = grad_col + b * col_channels * H_out * W_out;
				float* grad_image_b = grad_image + b * C * H * W;
				for ( int c = 0; c < C; ++c )
					for ( int kh = 0; kh < kH; ++kh )
						for ( int kw = 0; kw < kW; ++kw )
						{
							int col_row = c * kH * kW + kh * kW + kw;
							for ( int oh = 0; oh < H_out; ++oh )
								for ( int ow = 0; ow < W_out; ++ow )
								{
									int ih = oh * stride - pad + kh;
									int iw = ow * stride - pad + kw;
									if ( ih >= 0 && ih < H && iw >= 0 && iw < W )
									{
										int col_idx = col_row * (H_out * W_out) + oh * W_out + ow;
										grad_image_b[c * H * W + ih * W + iw] += grad_col_b[col_idx];
									}
								}
						}
			}
		};
	}
	return result;
}

} // namespace core
} // namespace tinytensor
