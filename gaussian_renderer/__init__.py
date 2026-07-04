# 角色：Python 高层渲染入口
# 它对应：
# gaussian_renderer/init.py 很像

import torch
# import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh

def renderTest():

    N = 1 << 25

    a1 = torch.arange(N, dtype=torch.float32, device="cuda")
    a2 = torch.arange(N, dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        a1=a1, 
        a2=a2
    )
    
    # 创建高斯光栅器
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 调用
    result = rasterizer()

    return result