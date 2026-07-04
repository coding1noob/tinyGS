
# NamedTuple 用来定义一种：“像 tuple 一样轻量，但每个位置有名字”的数据结构
from typing import NamedTuple
# 导入 torch.nn 这个子模块，并给它起别名 nn，不然那 nn.Module 要写成 torch.nn.Module
import torch.nn as nn
import torch

from . import _C

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, a2):
        args = (a1, a2)
        result = _C.mul_cuda(*args)
        return result

class GaussianRasterizationSettings(NamedTuple):
    a1: torch.Tensor
    a2: torch.Tensor

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self):
        return _RasterizeGaussians.apply(self.raster_settings.a1, self.raster_settings.a2)