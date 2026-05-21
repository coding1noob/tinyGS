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

// P：一共有多少个 Gaussian

// means3D：所有高斯中心的 3D 世界坐标数组，比如有三个高斯float means3D[] = {
//     0.0f,  0.0f,  3.0f,
//     0.5f,  0.0f,  3.2f,
//    -0.5f,  0.2f,  2.8f
// };

// scales：每个 Gaussian 在自己局部 3 个轴方向上的尺度

// float scales[] = {
//     0.15f, 0.15f, 0.15f,
//     0.20f, 0.10f, 0.10f,
//     0.12f, 0.18f, 0.12f
// };

// scale_modifier：对所有的高斯统一乘一个系数

// rotations：四元数

// float rotations[] = {
//     1.0f, 0.0f, 0.0f, 0.0f,
//     1.0f, 0.0f, 0.0f, 0.0f,
//     1.0f, 0.0f, 0.0f, 0.0f
// };

// opacities：每个高斯的基础不透明度

// float opacities[] = {
//     0.8f, 0.7f, 0.9f
// };

// shs：球谐颜色系数（Spherical Harmonics）

// sh_degree：SH 的阶数

// colors_precomp：每个 Gaussian 预先给好的 RGB 颜色

// cov3D_precomp：如果已经提前算好了每个 Gaussian 的 3D covariance，就从这里直接传进来；不过一般为空，传nullptr

// viewmatrix：世界坐标 → 相机坐标 的 4x4 视图矩阵

// projmatrix：4x4 投影矩阵，把 3D 点投到裁剪空间 / NDC 前

// cam_pos：相机在世界坐标下的位置

// tan_fovx：水平方向半视场角的正切值，比如如果水平 FOV 是 90°，那半角是 45°，所以：tan_fovx = 1.0f;

// W, H：输出图像宽高：background：背景颜色；out_color：渲染的颜色，3 * W * H

// out_invdepth：如果你想同时输出 inverse depth 图，就传一个长度 W * H 的数组

int renderCPU(
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
