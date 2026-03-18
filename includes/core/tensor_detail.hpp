#ifndef CVMML_CORE_TENSOR_DETAIL_HPP
#define CVMML_CORE_TENSOR_DETAIL_HPP

#include "core/tensor.hpp"
#include <cstdint>
#include <vector>

namespace cvmml {
namespace core {
namespace detail {

std::vector<int> make_contiguous_strides(const std::vector<int>& shape);
int64_t product_of(const std::vector<int>& values);
std::vector<int> unravel_index(int64_t linear, const std::vector<int>& shape);
int64_t ravel_index(const std::vector<int>& indices, const std::vector<int>& shape);
std::vector<int> broadcast_batch_shape(const std::vector<int>& a, const std::vector<int>& b);
int64_t map_broadcast_batch_index(const std::vector<int>& out_batch_index, const std::vector<int>& out_batch_shape, const std::vector<int>& operand_batch_shape);
std::vector<int> broadcast_shape_nd(const std::vector<int>& a, const std::vector<int>& b);
std::vector<int> normalize_axes(const std::vector<int>& axes, int rank);
bool has_axis(const std::vector<int>& axes, int axis);
std::vector<int> reduced_shape_from_axes(const std::vector<int>& in_shape, const std::vector<int>& axes, bool keepdim);
std::vector<int> to_keepdim_coords(const std::vector<int>& in_coords, const std::vector<int>& axes);
std::vector<int> to_reduced_coords(const std::vector<int>& keepdim_coords, const std::vector<int>& axes);
int64_t linear_for_broadcast_operand(const std::vector<int>& out_coords, const std::vector<int>& operand_shape, const std::vector<int>& operand_strides);
void check_same_device(const Tensor& a, const Tensor& b);

} // namespace detail
} // namespace core
} // namespace cvmml

#endif // CVMML_CORE_TENSOR_DETAIL_HPP
