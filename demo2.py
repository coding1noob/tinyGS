import time
import torch

from gaussian_renderer import renderTest

if __name__ == "__main__":
    N = 1 << 25

    # CUDA 计时
    torch.cuda.synchronize()
    t0 = time.time()
    cuda_result = renderTest()
    torch.cuda.synchronize()
    t1 = time.time()

    print(f"cuda计算结果: {cuda_result}")
    print(f"cuda用时: {t1 - t0:.6f} 秒")

    # CPU 计时
    common_result = 0.0
    t2 = time.time()
    for i in range(N):
        common_result += i * i
    t3 = time.time()

    print(f"不用cuda计算结果: {common_result}")
    print(f"不用cuda用时: {t3 - t2:.6f} 秒")
