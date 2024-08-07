#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= feats.size(0) || f >= feats.size(2))
        return;

    // point -1~1
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;

    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c;
    feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] +
                                   b * feats[n][1][f] +
                                   c * feats[n][2][f] +
                                   d * feats[n][3][f]) +
                        u * (a * feats[n][4][f] +
                             b * feats[n][5][f] +
                             c * feats[n][6][f] +
                             d * feats[n][7][f]);
}

torch::Tensor trilinear_fw_cu(
    const torch::Tensor feats,
    const torch::Tensor points)
{
    const int N = feats.size(0), F = feats.size(2);

    torch::Tensor feat_interp = torch::empty({N, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",
                               ([&]
                                { trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
                                      feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                      points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));

    return feat_interp;
}

template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_feat_dxyz)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= feats.size(0) || f >= feats.size(2))
        return;

    // point -1~1
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;

    scalar_t a = (1 - v) * (1 - w);
    scalar_t b = (1 - v) * w;
    scalar_t c = v * (1 - w);
    scalar_t d = 1 - a - b - c;

    dL_dfeats[n][0][f] = (1 - u) * a * dL_dfeat_interp[n][f];
    dL_dfeats[n][1][f] = (1 - u) * b * dL_dfeat_interp[n][f];
    dL_dfeats[n][2][f] = (1 - u) * c * dL_dfeat_interp[n][f];
    dL_dfeats[n][3][f] = (1 - u) * d * dL_dfeat_interp[n][f];
    dL_dfeats[n][4][f] = u * a * dL_dfeat_interp[n][f];
    dL_dfeats[n][5][f] = u * b * dL_dfeat_interp[n][f];
    dL_dfeats[n][6][f] = u * c * dL_dfeat_interp[n][f];
    dL_dfeats[n][7][f] = u * d * dL_dfeat_interp[n][f];

    // u
    dL_feat_dxyz[n][0][f] = (-(a * feats[n][0][f] +
                               b * feats[n][1][f] +
                               c * feats[n][2][f] +
                               d * feats[n][3][f]) +
                             (a * feats[n][4][f] +
                              b * feats[n][5][f] +
                              c * feats[n][6][f] +
                              d * feats[n][7][f])) *
                            dL_dfeat_interp[n][f];

    a = (1 - u) * (1 - w);
    b = (1 - u) * w;
    c = u * (1 - w);
    d = 1 - a - b - c;
    // v
    dL_feat_dxyz[n][1][f] = (-(a * feats[n][0][f] +
                               b * feats[n][1][f] +
                               c * feats[n][4][f] +
                               d * feats[n][5][f]) +
                             (a * feats[n][2][f] +
                              b * feats[n][3][f] +
                              c * feats[n][6][f] +
                              d * feats[n][7][f])) *
                            dL_dfeat_interp[n][f];

    a = (1 - u) * (1 - v);
    b = (1 - u) * v;
    c = u * (1 - v);
    d = 1 - a - b - c;
    // w
    dL_feat_dxyz[n][2][f] = (-(a * feats[n][0][f] +
                               b * feats[n][2][f] +
                               c * feats[n][4][f] +
                               d * feats[n][6][f]) +
                             (a * feats[n][1][f] +
                              b * feats[n][3][f] +
                              c * feats[n][5][f] +
                              d * feats[n][7][f])) *
                            dL_dfeat_interp[n][f];
}

template <typename scalar_t>
__global__ void dL_feat_dxyz_reduce_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_feat_dxyz,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dxyz)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= feats.size(0))
        return;

    for (int f = 0; f < feats.size(2); ++f)
    {
        dL_dxyz[n][0] += dL_feat_dxyz[n][0][f];
        dL_dxyz[n][1] += dL_feat_dxyz[n][1][f];
        dL_dxyz[n][2] += dL_feat_dxyz[n][2][f];
    }
    dL_dxyz[n][0] = dL_dxyz[n][0] / 2;
    dL_dxyz[n][1] = dL_dxyz[n][1] / 2;
    dL_dxyz[n][2] = dL_dxyz[n][2] / 2;

}

void trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points,
    torch::Tensor dL_dfeats,
    torch::Tensor dL_dxyz)
{
    const int N = feats.size(0), F = feats.size(2);

    // torch::Tensor dL_dfeats = torch::empty({N, 8, F}, feats.options());
    torch::Tensor dL_feat_dxyz = torch::empty({N, 3, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_bw_cu",
                               ([&]
                                { trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
                                      dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                      points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      dL_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                      dL_feat_dxyz.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()); }));

    const dim3 threads_reduce(16);
    const dim3 blocks_reduce((N + threads_reduce.x - 1) / threads_reduce.x);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "dL_feat_dxyz_reduce_cu",
                               ([&]
                                { dL_feat_dxyz_reduce_kernel<scalar_t><<<blocks_reduce, threads_reduce>>>(
                                      feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                      dL_feat_dxyz.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                      dL_dxyz.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));
}
