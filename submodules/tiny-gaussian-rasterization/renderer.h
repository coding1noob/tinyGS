#pragma once

#include <vector>

struct GaussianState {
    float depth = 0.0f;
    float mean2D[2] = {0.0f, 0.0f};
    float conic[3] = {0.0f, 0.0f, 0.0f};
    float opacity = 0.0f;
    float color[3] = {0.0f, 0.0f, 0.0f};
    int radius = 0;
    bool visible = false;
};

int rasterize_gaussians_cpu(
    int P,
    const float* means3D,
    const float* scales,
    float scale_modifier,
    const float* rotations,
    const float* opacities,
    const float* shs,
    int sh_degree,
    const float* colors_precomp,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    float tan_fovx,
    float tan_fovy,
    int W,
    int H,
    const float* background,
    float* out_color,
    float* out_invdepth = nullptr);
