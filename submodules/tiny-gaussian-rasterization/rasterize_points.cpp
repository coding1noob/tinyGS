#include "rasterize_points.h"
#include "renderer.h"

#include <stdexcept>
#include <string>

namespace {

void require_ndim(const py::buffer_info& info, int expected_ndim, const char* name) {
    if (info.ndim != expected_ndim) {
        throw std::runtime_error(std::string(name) + " must have " + std::to_string(expected_ndim) + " dimensions");
    }
}

void require_last_dim(const py::buffer_info& info, py::ssize_t expected, const char* name) {
    if (info.shape.back() != expected) {
        throw std::runtime_error(std::string(name) + " has wrong trailing dimension");
    }
}

void require_length(const py::buffer_info& info, py::ssize_t expected, const char* name) {
    if (info.size != expected) {
        throw std::runtime_error(std::string(name) + " has wrong number of elements");
    }
}

}  // namespace

py::tuple RenderCPUFromNumpy(
    py::array_t<float, py::array::c_style | py::array::forcecast> means3D,
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    float scale_modifier,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotations,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::object shs_obj,
    int sh_degree,
    py::object colors_precomp_obj,
    py::object cov3D_precomp_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast> viewmatrix,
    py::array_t<float, py::array::c_style | py::array::forcecast> projmatrix,
    py::array_t<float, py::array::c_style | py::array::forcecast> cam_pos,
    float tan_fovx,
    float tan_fovy,
    int width,
    int height,
    py::array_t<float, py::array::c_style | py::array::forcecast> background,
    bool return_invdepth)
{
    auto means_info = means3D.request();
    auto scales_info = scales.request();
    auto rotations_info = rotations.request();
    auto opacities_info = opacities.request();
    auto view_info = viewmatrix.request();
    auto proj_info = projmatrix.request();
    auto cam_info = cam_pos.request();
    auto bg_info = background.request();

    require_ndim(means_info, 2, "means3D");
    require_ndim(scales_info, 2, "scales");
    require_ndim(rotations_info, 2, "rotations");
    require_ndim(opacities_info, 1, "opacities");
    require_ndim(view_info, 2, "viewmatrix");
    require_ndim(proj_info, 2, "projmatrix");
    require_ndim(cam_info, 1, "cam_pos");
    require_ndim(bg_info, 1, "background");

    require_last_dim(means_info, 3, "means3D");
    require_last_dim(scales_info, 3, "scales");
    require_last_dim(rotations_info, 4, "rotations");
    require_length(cam_info, 3, "cam_pos");
    require_length(bg_info, 3, "background");

    if (view_info.shape[0] != 4 || view_info.shape[1] != 4) {
        throw std::runtime_error("viewmatrix must have shape (4, 4)");
    }
    if (proj_info.shape[0] != 4 || proj_info.shape[1] != 4) {
        throw std::runtime_error("projmatrix must have shape (4, 4)");
    }

    const int P = static_cast<int>(means_info.shape[0]);
    if (scales_info.shape[0] != P || rotations_info.shape[0] != P || opacities_info.shape[0] != P) {
        throw std::runtime_error("means3D/scales/rotations/opacities must agree on Gaussian count");
    }

    const float* shs_ptr = nullptr;
    if (!shs_obj.is_none()) {
        auto shs = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(shs_obj);
        shs_ptr = static_cast<const float*>(shs.request().ptr);
    }

    const float* colors_ptr = nullptr;
    if (!colors_precomp_obj.is_none()) {
        auto colors = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(colors_precomp_obj);
        auto colors_info = colors.request();
        require_ndim(colors_info, 2, "colors_precomp");
        require_last_dim(colors_info, 3, "colors_precomp");
        if (colors_info.shape[0] != P) {
            throw std::runtime_error("colors_precomp must have shape (P, 3)");
        }
        colors_ptr = static_cast<const float*>(colors_info.ptr);
    }

    const float* cov3D_ptr = nullptr;
    if (!cov3D_precomp_obj.is_none()) {
        auto cov3D = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cov3D_precomp_obj);
        auto cov3D_info = cov3D.request();
        require_ndim(cov3D_info, 2, "cov3D_precomp");
        require_last_dim(cov3D_info, 6, "cov3D_precomp");
        if (cov3D_info.shape[0] != P) {
            throw std::runtime_error("cov3D_precomp must have shape (P, 6)");
        }
        cov3D_ptr = static_cast<const float*>(cov3D_info.ptr);
    }

    py::array_t<float> out_color({3, height, width});
    auto out_color_info = out_color.request();

    py::array_t<float> out_invdepth;
    float* out_invdepth_ptr = nullptr;
    if (return_invdepth) {
        out_invdepth = py::array_t<float>({height, width});
        out_invdepth_ptr = static_cast<float*>(out_invdepth.request().ptr);
    }

    int visible = renderCPU(
        P,
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(scales_info.ptr),
        scale_modifier,
        static_cast<const float*>(rotations_info.ptr),
        static_cast<const float*>(opacities_info.ptr),
        shs_ptr,
        sh_degree,
        colors_ptr,
        cov3D_ptr,
        static_cast<const float*>(view_info.ptr),
        static_cast<const float*>(proj_info.ptr),
        static_cast<const float*>(cam_info.ptr),
        tan_fovx,
        tan_fovy,
        width,
        height,
        static_cast<const float*>(bg_info.ptr),
        static_cast<float*>(out_color_info.ptr),
        out_invdepth_ptr);

    if (return_invdepth) {
        return py::make_tuple(visible, out_color, out_invdepth);
    }
    return py::make_tuple(visible, out_color);
}
