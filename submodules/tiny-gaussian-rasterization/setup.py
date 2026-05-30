# 把 ext.cpp + rasterize_points.cpp + renderer.cpp 编译成一个 Python 模块，模块名就是你在 Extension(...) 里写的名字

from setuptools import Extension, setup

import pybind11

# setup.py 会把这些源码一起编译成一个叫 tiny_gaussian_rasterization 的 Python 扩展模块
ext_modules = [
    Extension(
        "tiny_gaussian_rasterization",
        sources=[
            "ext.cpp",
            "rasterize_points.cpp",
            "renderer.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "third_party/glm",
            ".",
        ],
        language="c++",
        extra_compile_args=["-std=c++20"],
    )
]

setup(
    name="tiny_gaussian_rasterization",
    version="0.0.1",
    ext_modules=ext_modules,
)
