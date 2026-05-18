// 角色：真正的 CPU 渲染实现
// 这是整个底层模块的核心计算文件

#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <vector>

// 先写help函数，就是辅助函数。意思是：
// 它不是整个渲染器的主入口
// 但主流程里很多地方都会反复用到它
// 所以单独拿出来写，避免主逻辑里塞一堆底层细节
// 下面三个原函数是在submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h里的
// 同时 namespace 后面不加名字，直接匿名 namespace，只给当前这个 .cpp 文件自己用，不暴露给别的

namespace
{
    // 原：__forceinline__ __device__ float ndc2Pix(float v, int S)
    // __device__表示GPU执行，
    // __forceinline__ 表示：强烈要求编译器把这个函数内联展开。普通内联为：inline
    // 作用：NDC 坐标 转成 像素坐标
    // NDC = Normalized Device Coordinates，也就是标准化设备坐标。
    // 在图形学里，做完投影除法后，一个点通常会落到类似这个范围：
    // x 在 [-1, 1], y 在 [-1, 1]
    // 输入：v 是 NDC 坐标，范围大致 [-1, 1], S 是图像尺寸，比如宽度或高度
    inline float ndc2Pix(float v, int S) 
    {
        return ((v + 1.0f) * S - 1.0f) * 0.5f;
    }

    // 原：__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
    // 然后把 float3 换成 glm::vec3
    // 作用：把一个 3D 点 用一个 4x4 变换矩阵的前 3 行 进行变换，返回一个 float3。你可以把它理解成：
    // “拿 3D 点做刚体/仿射变换，得到新的 3D 点”
    inline glm::vec3 transformPoint4x3(const glm::vec3& p, const float* matrix)
    {
        glm::vec3 transformed = {
            matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
            matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
            matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        };
        return transformed;
    }

    // 原：__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
    // 作用：它是把一个 3D 点 乘一个 完整 4x4 矩阵，得到一个 齐次坐标 float4
    inline glm::vec4 transformPoint4x4(const glm::vec3& p, const float* matrix)
    {
        glm::vec4 transformed = {
            matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
            matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
            matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
            matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
        };
        return transformed;
    }
}

// submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
// 原：__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
// scale:高斯在 3 个轴上的尺寸; mod:对高斯尺度做统一倍率调整的系数; rot:旋转参数,四元数字; cov3D:为3D协方差矩阵的压缩表示
void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// S 为 尺度
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// 把四元数拆成四个分量
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// 把四元数转成一个 3x3 旋转矩阵 R
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

    // 高斯局部主轴上的尺度 S, 和高斯的朝向 R, 组合起来
	glm::mat3 M = S * R;

	// 构造一个对称正定矩阵, 也就是合法的协方差矩阵, 直接优化 3x3 矩阵容易出非法值
	glm::mat3 Sigma = glm::transpose(M) * M;

	// 最后构造3D协方差矩阵
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}