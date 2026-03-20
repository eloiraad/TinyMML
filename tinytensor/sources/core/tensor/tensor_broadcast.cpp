#include "tensor_detail.hpp"
#include <algorithm>

namespace tinytensor {
namespace core {
namespace detail {

std::vector<int> make_contiguous_strides(const std::vector<int>& shape)
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

int64_t product_of(const std::vector<int>& values)
{
	int64_t product = 1;
	for ( int value : values )
		product *= value;
	return product;
}

std::vector<int> unravel_index(int64_t linear, const std::vector<int>& shape)
{
	std::vector<int> out(shape.size(), 0);
	for ( int i = static_cast<int>(shape.size()) - 1; i >= 0; --i )
	{
		out[i] = static_cast<int>(linear % shape[i]);
		linear /= shape[i];
	}
	return out;
}

int64_t ravel_index(const std::vector<int>& indices, const std::vector<int>& shape)
{
	if ( shape.empty() )
		return 0;
	int64_t linear = 0;
	for ( size_t i = 0; i < shape.size(); ++i )
		linear = linear * shape[i] + indices[i];
	return linear;
}

std::vector<int> broadcast_batch_shape(const std::vector<int>& a, const std::vector<int>& b)
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

int64_t map_broadcast_batch_index(const std::vector<int>& out_batch_index, const std::vector<int>& out_batch_shape, const std::vector<int>& operand_batch_shape)
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

std::vector<int> broadcast_shape_nd(const std::vector<int>& a, const std::vector<int>& b)
{
	size_t out_rank = std::max(a.size(), b.size());
	std::vector<int> out(out_rank, 1);

	for ( size_t i = 0; i < out_rank; ++i )
	{
		int a_dim = (i < out_rank - a.size()) ? 1 : a[i - (out_rank - a.size())];
		int b_dim = (i < out_rank - b.size()) ? 1 : b[i - (out_rank - b.size())];
		if ( a_dim != b_dim && a_dim != 1 && b_dim != 1 )
			throw std::invalid_argument("Shapes are not broadcast-compatible.");
		out[i] = std::max(a_dim, b_dim);
	}
	return out;
}

std::vector<int> normalize_axes(const std::vector<int>& axes, int rank)
{
	std::vector<int> out;
	if ( axes.empty() )
	{
		out.resize(rank);
		for ( int i = 0; i < rank; ++i )
			out[i] = i;
		return out;
	}

	for ( int axis : axes )
	{
		int normalized = axis;
		if ( normalized < 0 )
			normalized += rank;
		if ( normalized < 0 || normalized >= rank )
			throw std::invalid_argument("Axis out of range.");
		if ( std::find(out.begin(), out.end(), normalized) == out.end() )
			out.push_back(normalized);
	}

	std::sort(out.begin(), out.end());
	return out;
}

bool has_axis(const std::vector<int>& axes, int axis)
{
	return std::find(axes.begin(), axes.end(), axis) != axes.end();
}

std::vector<int> reduced_shape_from_axes(const std::vector<int>& in_shape, const std::vector<int>& axes, bool keepdim)
{
	std::vector<int> out;
	for ( int i = 0; i < static_cast<int>(in_shape.size()); ++i )
	{
		if ( has_axis(axes, i) )
		{
			if ( keepdim )
				out.push_back(1);
		}
		else
			out.push_back(in_shape[i]);
	}
	if ( out.empty() )
		out.push_back(1);
	return out;
}

std::vector<int> to_keepdim_coords(const std::vector<int>& in_coords, const std::vector<int>& axes)
{
	std::vector<int> out = in_coords;
	for ( int axis : axes )
		out[axis] = 0;
	return out;
}

std::vector<int> to_reduced_coords(const std::vector<int>& keepdim_coords, const std::vector<int>& axes)
{
	std::vector<int> out;
	for ( int i = 0; i < static_cast<int>(keepdim_coords.size()); ++i )
		if ( !has_axis(axes, i) )
			out.push_back(keepdim_coords[i]);
	if ( out.empty() )
		out.push_back(0);
	return out;
}

int64_t linear_for_broadcast_operand(const std::vector<int>& out_coords, const std::vector<int>& operand_shape, const std::vector<int>& operand_strides)
{
	int out_rank = static_cast<int>(out_coords.size());
	int in_rank = static_cast<int>(operand_shape.size());
	int shift = out_rank - in_rank;
	int64_t linear = 0;
	for ( int i = 0; i < in_rank; ++i )
	{
		int coord = out_coords[i + shift];
		if ( operand_shape[i] == 1 )
			coord = 0;
		linear += static_cast<int64_t>(coord) * operand_strides[i];
	}
	return linear;
}

void check_same_device(const Tensor& a, const Tensor& b)
{
	if ( a.device() != b.device() )
		throw std::invalid_argument("Tensors must be on the same device for this operation.");
}

} // namespace detail
} // namespace core
} // namespace tinytensor
