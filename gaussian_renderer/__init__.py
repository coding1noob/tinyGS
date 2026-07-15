# 角色：Python 高层渲染入口
# 它对应：
# gaussian_renderer/init.py 很像

import torch
# import math

# 导入 submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py中的类
from diff_gaussian_rasterization import MultiplySettings, Multiply

# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh

def MultiplyTest():
    N = 1 << 25

    a1 = torch.arange(N, dtype=torch.float32, device="cuda")
    a2 = torch.arange(N, dtype=torch.float32, device="cuda")

    multiply_settings = MultiplySettings(
        a1=a1,      # 此处仅为模仿
        a2=a2
    )
    
    # 创建高斯光栅器
    Multiplyer = Multiply(multiply_settings=multiply_settings)

    # 因为 GaussianRasterizer类 是 nn.Module 的基类，所以调用此 rasterizer 等于 rasterizer.forward(...)
    result = Multiplyer()

    return result