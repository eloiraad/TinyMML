#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include "core/tensor.hpp"

namespace py = pybind11;
using namespace cvmml::core;

PYBIND11_MODULE(cvmml_api, m)
{
	m.doc() = "CVMML: Computer Vision and Machine Learning Library C++ Engine";

	py::enum_<Device>(m, "Device")
		.value("CPU", Device::CPU)
		.value("CUDA", Device::CUDA)
		.export_values();

	py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
		.def(py::init<const std::vector<int>&>(), py::arg("shape"))

		.def("shape", &Tensor::shape)
		.def("strides", &Tensor::strides)
		.def("size", &Tensor::size)
		.def("is_view", &Tensor::is_view)
		.def("is_contiguous", &Tensor::is_contiguous)
		.def("contiguous", &Tensor::contiguous)
		.def("view", &Tensor::view, py::arg("new_shape"))

		.def("requires_grad", &Tensor::requires_grad)
		.def("set_requires_grad", &Tensor::set_requires_grad, py::arg("val"))
		.def("zero_grad", &Tensor::zero_grad)
		.def("backward", &Tensor::backward)
		.def("grad", [](const Tensor& t) -> py::object
		{
			if ( !t.grad() )
				return py::none();
			std::vector<py::ssize_t> numpy_strides(t.strides().size());
			for ( size_t i = 0; i < t.strides().size(); ++i )
				numpy_strides[i] = t.strides()[i] * sizeof(float);
			return py::array(py::buffer_info(
				t.grad(),
				sizeof(float),
				py::format_descriptor<float>::format(),
				t.shape().size(),
				t.shape(),
				numpy_strides
			));
		})

		.def(py::self + py::self)
		.def(py::self - py::self)
		.def(py::self * py::self)
		.def(py::self / py::self)

		.def(py::self + float())
		.def(py::self - float())
		.def(py::self * float())
		.def(py::self / float())
		.def("exp", &Tensor::exp)
		.def("log", &Tensor::log)
		.def("sum", &Tensor::sum, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
		.def("max", &Tensor::max, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
		.def("min", &Tensor::min, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)

		.def("mm", &Tensor::matmult)
		.def("__matmul__", &Tensor::matmult)

		.def("T", &Tensor::transpose, py::arg("dim0") = -2, py::arg("dim1") = -1)
		.def_static("zeros", &Tensor::zeros, py::arg("shape"))
		.def_static("ones", &Tensor::ones, py::arg("shape"))
		.def_static("randn", &Tensor::randn, py::arg("shape"), py::arg("mean")=0.0f, py::arg("std")=1.0f)
		.def_static("bernoulli", &Tensor::bernoulli, py::arg("shape"), py::arg("p"))
		.def_static("uniform", &Tensor::uniform, py::arg("shape"))

		.def("im2col", &Tensor::im2col, py::arg("kH"), py::arg("kW"), py::arg("stride")=1, py::arg("pad")=0)

		.def("__repr__", [](const Tensor& t)
		{
			py::module_ np = py::module_::import("numpy");
			py::object np_array = py::array(py::cast(t));
			std::string np_str = py::str(np_array);
			return "Tensor(shape=[" + py::str(py::cast(t.shape())).cast<std::string>() + "])\n" + np_str;
		})

		.def_buffer([](Tensor& t) -> py::buffer_info
		{
			std::vector<py::ssize_t> numpy_strides(t.strides().size());
			for ( size_t i = 0; i < t.strides().size(); ++i )
				numpy_strides[i] = t.strides()[i] * sizeof(float);
			return py::buffer_info(
				t.data(),
				sizeof(float),
				py::format_descriptor<float>::format(),
				t.shape().size(),
				t.shape(),
				numpy_strides
			);
		})

		.def("device", &Tensor::device)
		.def("to_cuda", &Tensor::to_cuda)
		.def("to_cpu", &Tensor::to_cpu)
		.def("relu", &Tensor::relu)
		.def("sqrt", &Tensor::sqrt)
		.def("mean", &Tensor::mean, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
		.def("var", &Tensor::var, py::arg("axes") = std::vector<int>{}, py::arg("keepdim") = false)
		.def("subtract_", &Tensor::subtract_)
		.def("__isub__", [](Tensor& self, const Tensor& other) -> Tensor& {
			self.subtract_(other);
			return self;
		}, py::return_value_policy::reference_internal);
}