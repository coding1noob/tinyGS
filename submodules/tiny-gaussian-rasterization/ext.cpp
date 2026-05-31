#include "rasterize_points.h"

// 当这个扩展模块被 Python 成功 import 之后，它里面会有一个函数叫 render_cpu
// 这里和3dgs不同，因为编译出来的子模块名就是_C
PYBIND11_MODULE(_C, m) 
{
    m.def("render_cpu", &RenderCPUFromNumpy,
          py::arg("means3D"),
          py::arg("scales"),
          py::arg("scale_modifier"),
          py::arg("rotations"),
          py::arg("opacities"),
          py::arg("shs") = py::none(),
          py::arg("sh_degree") = 0,
          py::arg("colors_precomp") = py::none(),
          py::arg("cov3D_precomp") = py::none(),
          py::arg("viewmatrix"),
          py::arg("projmatrix"),
          py::arg("cam_pos"),
          py::arg("tan_fovx"),
          py::arg("tan_fovy"),
          py::arg("width"),
          py::arg("height"),
          py::arg("background"),
          py::arg("return_invdepth") = false);
}
