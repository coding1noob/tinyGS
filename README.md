一开始只有main.cpp，CMakeLists.txt，tgaimage.cpp，tgaimage.h

然后新建了submodules/tiny-gaussian-rasterization里面的东西

同时把原版高斯的glm库复制进来

先写renderer.cpp：

    第一步
    写 helper + computeCov3D

    第二步
    写 computeCov2D

    第三步
    写 preprocessGaussians

    第四步
    写 rasterize_gaussians_cpu

前四步完成后，在main.cpp中做一个小的闭环！

第五步
再考虑怎么接 Python

    原项目中：
        先看 3dgs_note/submodules/diff-gaussian-rasterization/ext.cpp                       
        负责把 C++/CUDA 函数暴露给 Python   （定义）

        再看 3dgs_note/submodules/diff-gaussian-rasterization/rasterize_points.cu           
        负责把输入整理后调用渲染核心          （调用）

        再看 3dgs_note/submodules/diff-gaussian-rasterization/setup.py                      
        负责把上面这些源码（原文是rasterize_points.cu、ext.cpp、cuda_rasterizer/forward.cu、cuda_rasterizer/backward.cu等等这些）编译成 Python 能 import 的扩展模块

        最后 3dgs_note/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py   
        import 上面的扩展并包装它