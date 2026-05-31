from . import _C

# 这里相当与做了层包装，外面调用就用 from tiny_gaussian_rasterization import render_cpu
# 让用户优先接触 render_cpu(...)，而不是直接接触底层 _C
def render_cpu(
    means3D,
    scales,
    scale_modifier,
    rotations,
    opacities,
    shs=None,
    sh_degree=0,
    colors_precomp=None,
    cov3D_precomp=None,
    viewmatrix=None,
    projmatrix=None,
    cam_pos=None,
    tan_fovx=1.0,
    tan_fovy=1.0,
    width=512,
    height=512,
    background=None,
    return_invdepth=False,
):
    return _C.render_cpu(
        means3D,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        sh_degree,
        colors_precomp,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,
        tan_fovx,
        tan_fovy,
        width,
        height,
        background,
        return_invdepth,
    )
