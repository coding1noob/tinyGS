# 角色：最外层使用者 / 测试脚本
# 此 python 可理解成“你手动扮演 render.py 或 train.py 的一个极简版”

import numpy as np  # 导入 numpy 这个模块，并给它起一个简短别名 np

from tiny_gaussian_rasterization import render_cpu

def main():
    W = 512
    H = 512
    P = 3

    means3D = np.array([
        [ 0.0, 0.0, 3.0],
        [ 0.6, 0.0, 3.2],
        [-0.5, 0.3, 2.8],
    ], dtype=np.float32)

    scales = np.array([
        [0.18, 0.18, 0.18],
        [0.15, 0.12, 0.15],
        [0.14, 0.20, 0.14],
    ], dtype=np.float32)

    scale_modifier = 3.0

    rotations = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    opacities = np.array([
        0.90, 0.75, 0.85
    ], dtype=np.float32)

    shs = None
    sh_degree = 0

    colors_precomp = np.array([
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.2, 0.3, 1.0],
    ], dtype=np.float32)

    cov3D_precomp = None

    cam_pos = np.array([
        0.0, 0.0, 0.0
    ], dtype=np.float32)

    viewmatrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    tan_fovx = 1.0
    tan_fovy = 1.0

    projmatrix = np.array([
        [1.0 / tan_fovx, 0.0, 0.0,  0.0],
        [0.0, 1.0 / tan_fovy, 0.0,  0.0],
        [0.0, 0.0, 1.0,  1.0],
        [0.0, 0.0, -0.2, 0.0],
    ], dtype=np.float32)

    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # out_color 里面其实已经是整张图了
    visible, out_color = render_cpu(
    means3D=means3D,
    scales=scales,
    scale_modifier=scale_modifier,
    rotations=rotations,
    opacities=opacities,
    shs=shs,
    sh_degree=sh_degree,
    colors_precomp=colors_precomp,
    cov3D_precomp=cov3D_precomp,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    cam_pos=cam_pos,
    tan_fovx=tan_fovx,
    tan_fovy=tan_fovy,
    width=W,
    height=H,
    background=background,
    return_invdepth=False,
    )

    print("visible =", visible)
    print("out_color.shape =", out_color.shape)
    print("out_color.dtype =", out_color.dtype)
    print("out_color.min =", out_color.min())
    print("out_color.max =", out_color.max())

    # 因为你 C++ 输出的 out_color 形状是：(3, H, W)，但大多数 Python 图像库想要的是：(H, W, 3)
    img = np.transpose(out_color, (1, 2, 0))
    # 限制范围在 0.0 到 1.0 之间
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    import imageio.v2 as imageio
    imageio.imwrite("output.png", img)
    print("saved output.png")


# 只有这个 .py 文件被直接运行时，条件才为真；如果只是被别的 Python 文件 import，就不会执行这块代码
if __name__ == "__main__":
    main()