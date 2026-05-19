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
	// 点 p = (x,y,z) 化为齐次坐标变为 (x,y,z,1)，然后 乘一个 4x4 仿射变换矩阵，
	// 但是只关注乘之后的前三个变量，所以叫 4x3
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
    // 和上面不同，需要 w 分量，来做 x/w, y/w, z/w 完整投影变换
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
	// S 为 缩放，因为一开始都是球
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

    // 高斯局部主轴上的缩放尺度 S, 和高斯的朝向 R, 组合起来
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

// 输入：mean：高斯中心（世界坐标），focal_x：x 方向像素焦距，tan_fovx：水平半视场角正切，cov3D：高斯 3D 协方差，viewmatrix：世界到相机的变换矩阵
// 输出：一个 glm::vec3，x = cov00，y = cov01，z = cov11
glm::vec3 computeCov2D(const glm::vec3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// 把高斯中心 mean 从世界坐标变到相机坐标，t.z：点距离相机前方多远
	glm::vec3 t = transformPoint4x3(mean, viewmatrix);

	// 1.3f 表示允许点稍微超出屏幕边缘一点
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = std::min(limx, std::max(-limx, txtz)) * t.z;
	t.y = std::min(limy, std::max(-limy, tytz)) * t.z;

	// J 和 W 就是论文中的公式（5）
	// J 描述的是：在当前高斯中心附近，如果 3D 点发生一个微小扰动，它在屏幕上会产生什么样的微小变化
	// 为什么要用 Jacobian？透视投影不是线性的，比如点离相机近和远，屏幕变化尺度完全不同。
	// 但在“一个非常小的局部范围内”，我们可以用线性近似，这个局部线性近似就是 Jacobian。
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 取的是 viewmatrix 左上角的 3x3 旋转/线性部分
	// 因为：协方差只关心局部形状变化，不关心平移，平移是 mean 决定的
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	
	// 理论上应该是 J*W，因为先 world -> camera，再 camera -> image 
	// 但是因为GLM的存储方式，以及作者注释里说的 row/column-major convention 问题写成 W*J
	// 总之就是 世界空间的小扰动 → 屏幕空间的小扰动
	glm::mat3 T = W * J;

	// Vrk 是 3D 协方差矩阵
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 高斯投影到屏幕空间后的 2D 协方差（这里暂时放在 3x3 形式里），本质上它是把 3D 协方差通过线性映射 T 推到 2D 屏幕空间
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// CPU equivalent of preprocessCUDA (forward.cu)
std::vector<GaussianState> preprocessGaussians(
    int P,
    const float* means3D,			// float类型的数组
    const float* scales,
    float scale_modifier,
    const float* rotations,
    const float* opacities,
    const float* shs,
    int sh_degree,
    const float* colors_precomp,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    float tan_fovx,
    float tan_fovy,
    int W,
    int H)
{
    const float focal_x = static_cast<float>(W) / (2.0f * tan_fovx);
    const float focal_y = static_cast<float>(H) / (2.0f * tan_fovy);

    std::vector<GaussianState> states(P);

    for (int idx = 0; idx < P; ++idx) {
		// 临时变量，最后用于 states[idx] = state
        GaussianState state{};

		// 世界坐标系下的高斯中心
        glm::vec3 mean = {
            means3D[3 * idx + 0],
            means3D[3 * idx + 1],
            means3D[3 * idx + 2]
        };

		// viewmatrix 代表 世界坐标 → 相机坐标，原文没直接求，封装在 in_frustum 里了
        glm::vec3 p_view = transformPoint4x3(mean, viewmatrix);
		// projmatrix 代表 透视投影
        glm::vec4 p_hom = transformPoint4x4(mean, projmatrix);

        float inv_w = 1.0f / (p_hom.w + 1e-7f);
        glm::vec3 p_proj = {
            p_hom.x * inv_w,
            p_hom.y * inv_w,
            p_hom.z * inv_w
        };

        // Corresponds to in_frustum / near culling
        if (p_view.z <= 0.2f) {
            states[idx] = state;
            continue;
        }

        float local_cov3D[6];
        const float* cov3D_ptr = nullptr;

        if (cov3D_precomp != nullptr) {
            cov3D_ptr = cov3D_precomp + idx * 6;
        } else {
            glm::vec3 scale = {
                scales[3 * idx + 0],
                scales[3 * idx + 1],
                scales[3 * idx + 2]
            };

            glm::vec4 rotation = {
                rotations[4 * idx + 0],
                rotations[4 * idx + 1],
                rotations[4 * idx + 2],
                rotations[4 * idx + 3]
            };

			// 有 scale，scale的统一缩放系数，rotation，足够构造协方差矩阵 local_cov3D 了
            computeCov3D(scale, scale_modifier, rotation, local_cov3D);
            cov3D_ptr = local_cov3D;
        }

        glm::vec3 cov = computeCov2D(
            mean,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
            cov3D_ptr,
            viewmatrix
        );

		// 在 2D 协方差的对角线上加一个小常数，避免高斯过尖、过小、数值不稳定
        constexpr float h_var = 0.3f;
        float det_cov = cov.x * cov.z - cov.y * cov.y;
        cov.x += h_var;
        cov.z += h_var;

        float det = cov.x * cov.z - cov.y * cov.y;
        if (det == 0.0f) {
            states[idx] = state;
            continue;
        }

        float det_inv = 1.0f / det;
        glm::vec3 conic = {
            cov.z * det_inv,
            -cov.y * det_inv,
            cov.x * det_inv
        };

		// 算屏幕半径 my_radius，这个之后可视化一下！
        float mid = 0.5f * (cov.x + cov.z);
        float lambda1 = mid + std::sqrt(std::max(0.1f, mid * mid - det));
        float lambda2 = mid - std::sqrt(std::max(0.1f, mid * mid - det));
        float my_radius = std::ceil(3.0f * std::sqrt(std::max(lambda1, lambda2)));

		// 把透视后的坐标变到符合屏幕大小的变换（视口变换）
        glm::vec2 point_image = {
            ndc2Pix(p_proj.x, W),
            ndc2Pix(p_proj.y, H)
        };

        if (my_radius <= 0.0f) {
            states[idx] = state;
            continue;
        }

        glm::vec3 color;
        if (colors_precomp != nullptr) {
            color = {
                colors_precomp[3 * idx + 0],
                colors_precomp[3 * idx + 1],
                colors_precomp[3 * idx + 2]
            };
        } else {
            color = {1.0f, 1.0f, 1.0f};
        }

        state.visible = true;
        state.depth = p_view.z;
        state.radius = static_cast<int>(my_radius);
        state.mean2D[0] = point_image.x;
        state.mean2D[1] = point_image.y;
        state.conic[0] = conic.x;
        state.conic[1] = conic.y;
        state.conic[2] = conic.z;
        state.opacity = opacities[idx];
        state.color[0] = color.x;
        state.color[1] = color.y;
        state.color[2] = color.z;

        states[idx] = state;
    }

    return states;
}
