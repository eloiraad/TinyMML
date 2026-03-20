#ifndef CVMML_CORE_CUDA_KERNELS_COMMON_CUH
#define CVMML_CORE_CUDA_KERNELS_COMMON_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) { \
	cudaError_t err = call; \
	if ( err != cudaSuccess ) \
		throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
}
#endif

#ifndef CHECK_CUDA_LAUNCH
#define CHECK_CUDA_LAUNCH() CHECK_CUDA(cudaPeekAtLastError())
#endif

namespace cvmml {
namespace core {
namespace cuda {

inline void get_grid_1d(int size, dim3& blockSize, dim3& gridSize)
{
	blockSize = dim3(256);
	gridSize = dim3((size + blockSize.x - 1) / blockSize.x);
}

} // namespace cuda
} // namespace core
} // namespace cvmml

#endif // CVMML_CORE_CUDA_KERNELS_COMMON_CUH
