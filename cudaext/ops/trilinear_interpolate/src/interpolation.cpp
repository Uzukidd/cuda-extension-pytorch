#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor trilinear_fw_cu(
    const torch::Tensor feats,
    const torch::Tensor points);

void trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points,
    torch::Tensor dL_dfeats,
    torch::Tensor dL_dxyz);

torch::Tensor trilinear_interpolation_fw(
    const torch::Tensor feats,
    const torch::Tensor points)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_fw_cu(feats, points);
}

void trilinear_interpolation_bw(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points,
    torch::Tensor dL_dfeats,
    torch::Tensor dL_dxyz)
{
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);
    CHECK_INPUT(dL_dfeats);
    CHECK_INPUT(dL_dxyz);

    trilinear_bw_cu(dL_dfeat_interp, feats, points, dL_dfeats, dL_dxyz);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("trilinear_interpolation_fw", &trilinear_interpolation_fw);
    m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw);
}
