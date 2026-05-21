#include "renderer.h"
#include "tgaimage.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int W = 512;
    const int H = 512;
    const int P = 3;

    // 3 个高斯中心，放在相机前方
    float means3D[3 * P] = 
    {
         0.0f,  0.0f,  3.0f,
         0.6f,  0.0f,  3.2f,
        -0.5f,  0.3f,  2.8f
    };

    // 每个高斯的 3D 尺度
    float scales[3 * P] = 
    {
        0.18f, 0.18f, 0.18f,
        0.15f, 0.12f, 0.15f,
        0.14f, 0.20f, 0.14f
    };

    float scale_modifier = 3.0f;

    // 单位四元数：不旋转
    float rotations[4 * P] = 
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f
    };

    // 每个高斯的基础 opacity
    float opacities[P] = 
    {
        0.90f, 0.75f, 0.85f
    };

    // 先不用 SH
    const float* shs = nullptr;
    int sh_degree = 0;

    // 直接给预计算颜色 RGB
    float colors_precomp[3 * P] = 
    {
        1.0f, 0.1f, 0.1f,   // red
        0.1f, 1.0f, 0.1f,   // green
        0.2f, 0.3f, 1.0f    // blue
    };

    // 先不传预计算 covariance，让 renderCPU 内部自己算
    const float* cov3D_precomp = nullptr;

    // 相机位置：原点
    float cam_pos[3] = {0.0f, 0.0f, 0.0f};

    // 最简单：view 先用单位矩阵
    float viewmatrix[16] = 
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    // 设一个简单 FOV
    const float tan_fovx = 1.0f;
    const float tan_fovy = 1.0f;

    // 和你当前 transformPoint4x4 的乘法方式兼容的一个简单投影矩阵
    // 目标只是先让 NDC 投影跑起来，不追求和 OpenGL/3DGS 完全一致
    float projmatrix[16] = {
        1.0f / tan_fovx, 0.0f,            0.0f,  0.0f,
        0.0f,            1.0f / tan_fovy, 0.0f,  0.0f,
        0.0f,            0.0f,            1.0f,  1.0f,
        0.0f,            0.0f,           -0.2f,  0.0f
    };

    float background[3] = {0.0f, 0.0f, 0.0f};

    std::vector<float> out_color(3 * W * H, 0.0f);
    std::vector<float> out_invdepth(W * H, 0.0f);

    int visible = renderCPU(
        P,
        means3D,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        sh_degree,
        colors_precomp,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,
        tan_fovx,
        tan_fovy,
        W,
        H,
        background,
        out_color.data(),
        out_invdepth.data()
    );

    std::cout << "Visible gaussians: " << visible << std::endl;

    // 渲染
    TGAImage framebuffer(W, H, TGAImage::RGB);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int pix_id = y * W + x;

            float r = out_color[0 * H * W + pix_id];
            float g = out_color[1 * H * W + pix_id];
            float b = out_color[2 * H * W + pix_id];

            r = std::clamp(r, 0.0f, 1.0f);
            g = std::clamp(g, 0.0f, 1.0f);
            b = std::clamp(b, 0.0f, 1.0f);

            unsigned char R = static_cast<unsigned char>(r * 255.0f);
            unsigned char G = static_cast<unsigned char>(g * 255.0f);
            unsigned char B = static_cast<unsigned char>(b * 255.0f);

            TGAColor color;
            color[0] = B;
            color[1] = G;
            color[2] = R;
            color[3] = 255;
            framebuffer.set(x, y, color);
        }
    }

    framebuffer.write_tga_file("output3.tga");
    std::cout << "Saved output.tga" << std::endl;

    return 0;
}
