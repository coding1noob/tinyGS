# 只依托 python, C++ 完成渲染

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
        import 上面的扩展并包装它。通过 ext.cpp 的定义，来调用 rasterize_points.cu 里面的函数

    写完之后从顶层来看调用链条就是：
        demo.py
        -> from tiny_gaussian_rasterization import render_cpu
        -> tiny_gaussian_rasterization/__init__.py
        -> from . import _C
        -> _C.render_cpu(...)
        -> RenderCPUFromNumpy(...)
        -> renderCPU(...)
        -> out_color


第五步写完之后测试闭环：

    submodules/tiny-gaussian-rasterization/
    目录下编译扩展：
    python setup.py build_ext --inplace
        (目前用的是上面这句而不是 pip install )
        上面这句话为本地开发测试，先把扩展编出来让我当前目录能 import      
        build_ext：只构建扩展；      -inplace：把编译结果直接放在源码包目录附近，而不是只放进临时 build 目录
        然后就能生成.so文件在tiny_gaussian_rasterization

        而 python -m pip install -e . 是正式把这个项目装进当前 Python 环境

    然后（当然如果执行的是 python -m pip install -e . 就不用前面的路径）：
    PYTHONPATH=submodules/tiny-gaussian-rasterization python demo.py