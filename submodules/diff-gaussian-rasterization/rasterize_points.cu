
// #include <math.h>
#include <torch/extension.h>
// #include <cstdio>
// #include <sstream>
// #include <iostream>
// #include <tuple>
// #include <stdio.h>
#include <cuda_runtime_api.h>
// #include <memory>
// #include "cuda_rasterizer/config.h"
// #include "cuda_rasterizer/rasterizer.h"
// #include <fstream>
// #include <string>
// #include <functional>

RasterizeGaussiansCUDA
(
    const torch::Tensor &means3D,
    const int image_height,
    const int image_width
)
{
    // 首先要保证 means3D 是二维,且第二维得为3
    if( means3D.ndimension()!=2 || means3D.size(1)!=3 )
        AT_ERROR("means3D形式必须是(num_points, 3)");

    // 基于 means3D 当前的 tensor 属性，准备两套 tensor 创建选项：一套 int32，一套 float32
    // int_opts就是一份“和 means3D 在同一个设备上，但数据类型是 int32”的 tensor 配置
    // options()表示"取 means3D 这个 tensor 的各种配置选项"
    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    
    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    torch::Tensor out_color = torch::full()

}



// ===================================== 乘法测试 =========================================

// __global__ 表示这是一个 kernel 函数，也就是：由 CPU 端发起调用，在 GPU 上由大量线程并行执行
__global__ void MultiplyKernel(const float* a1, const float* a2, float* out, int n) 
{
    // 当前线程的全局编号 = 它所在 block 的编号 × 每个 block 的线程数 + 它在该 block 内部的编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a1[idx] * a2[idx];
    }
}

torch::Tensor MultiplyCUDA(torch::Tensor a1, torch::Tensor a2) 
{
    auto out = torch::zeros_like(a1);
    int n = static_cast<int>(a1.numel());

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    MultiplyKernel<<<blocks, threads>>>
    (
        a1.data_ptr<float>(),
        a2.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    // 乘法是 kernel 做的，但是累加是调用 PyTorch tensor 做的
    return out.sum();
}
