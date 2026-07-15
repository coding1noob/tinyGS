# 负责把上面这些源码（原文是rasterize_points.cu、ext.cpp、cuda_rasterizer/forward.cu、
# cuda_rasterizer/backward.cu等等这些）编译成 Python 能 import 的扩展模块

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# name="diff_gaussian_rasterization" 是整个 Python 包的名字，pip install ... 时，安装的是这个项目名
# 在外面import . 或者import diff_gaussian_rasterization，就缘于这

# packages=['diff_gaussian_rasterization']
# 告诉 setuptools：这个项目里有一个 Python 包，叫 diff_gaussian_rasterization

# ext_modules=[         表示：
# 这个包里除了纯 Python 文件，还要编译一个或多个 C++/CUDA 扩展模块

# CUDAExtension(        表示这是一个 PyTorch 的 CUDA 扩展。
# 它会用 CUDA 工具链去编译 .cu 文件

# name="diff_gaussian_rasterization._C"
# 定义扩展模块的完整 Python 导入名。所以最终 Python 里你能写：
# from diff_gaussian_rasterization import _C 或 from . import _C 或 import diff_gaussian_rasterization._C

# sources=[             这些是编译这个扩展模块所需的源文件
setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)