
# NamedTuple 用来定义一种：“像 tuple 一样轻量，但每个位置有名字”的数据结构
from typing import NamedTuple
# 导入 torch.nn 这个子模块，并给它起别名 nn，不然那 nn.Module 要写成 torch.nn.Module
import torch.nn as nn
import torch

from . import _C

# class _RasterizeGaussians(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, a1, a2):
#         args = (a1, a2)
#         result = _C.mul_cuda(*args)
#         return result

# class GaussianRasterizationSettings(NamedTuple):
#     a1: torch.Tensor
#     a2: torch.Tensor

# class GaussianRasterizer(nn.Module):
#     def __init__(self, raster_settings):
#         super().__init__()
#         self.raster_settings = raster_settings

#     def forward(self):
#         # _RasterizeGaussians为自定义的类，但是继承了torch.autograd.Function。
#         # 当调用_RasterizeGaussians.apply的时候
#         # PyTorch 自动做：1. 调用 forward 得到输出；2. 把 forward 过程记录进计算图；
#         # 3. 保存 backward 需要的信息 ctx.save_for_backward(...)；
#         # 4. 当 .backward() 时自动调用 backward
#         return _RasterizeGaussians.apply(self.raster_settings.a1, self.raster_settings.a2)

class _Multiplys(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, a2):
        args = (a1, a2)
        result = _C.mul_cuda(*args)
        return result

class MultiplySettings(NamedTuple):
    a1: torch.Tensor
    a2: torch.Tensor

class Multiply(nn.Module):
    def __init__(self, multiply_settings):
        super().__init__()
        self.multiply_settings = multiply_settings

    def forward(self):
        # _Multiplys为自定义的类，但是继承了torch.autograd.Function。
        # 当调用 _Multiplys.apply 的时候
        # PyTorch 自动做：1. 调用 forward 得到输出；2. 把 forward 过程记录进计算图；
        # 3. 保存 backward 需要的信息 ctx.save_for_backward(...)；
        # 4. 当 .backward() 时自动调用 backward
        return _Multiplys.apply(self.multiply_settings.a1, self.multiply_settings.a2)