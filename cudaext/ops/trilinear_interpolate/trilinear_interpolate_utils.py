import torch
from . import trilinear_interpolate_cuda


class Trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points):
        """
        Inputs:
            feats: (N, 8, F)
            points: (N, 3) local coordinates in [-1, 1]

        Outputs:
            feats_interp: (N, F)
        """
        feat_interp = trilinear_interpolate_cuda.trilinear_interpolation_fw(
            feats, points)

        ctx.save_for_backward(feats, points)

        return feat_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp):
        feats, points = ctx.saved_tensors

        dL_dfeats = trilinear_interpolate_cuda.trilinear_interpolation_bw(
            dL_dfeat_interp.contiguous(), feats, points)

        return dL_dfeats, None


def trilinear_interpolation_cpu(feats, points):
    """
    Inputs:
        feats: (N, 8, F)
        points: (N, 3) local coordinates in [-1, 1]

    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
        u*(a*feats[:, 4] +
           b*feats[:, 5] +
           c*feats[:, 6] +
           d*feats[:, 7])

    return feats_interp
