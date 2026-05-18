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

第五步
再考虑怎么接 main.cpp 或 Python