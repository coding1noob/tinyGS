#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::tuple RenderCPUFromNumpy(
    py::array_t<float, py::array::c_style | py::array::forcecast> means3D,
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    float scale_modifier,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotations,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::object shs,
    int sh_degree,
    py::object colors_precomp,
    py::object cov3D_precomp,
    py::array_t<float, py::array::c_style | py::array::forcecast> viewmatrix,
    py::array_t<float, py::array::c_style | py::array::forcecast> projmatrix,
    py::array_t<float, py::array::c_style | py::array::forcecast> cam_pos,
    float tan_fovx,
    float tan_fovy,
    int width,
    int height,
    py::array_t<float, py::array::c_style | py::array::forcecast> background,
    bool return_invdepth);
