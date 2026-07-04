#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul_cuda", &MultiplyCUDA);
//   m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
//   m.def("mark_visible", &markVisible);
}