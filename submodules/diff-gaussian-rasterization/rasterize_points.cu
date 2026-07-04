
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
