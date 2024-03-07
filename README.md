# cuda-extension-pytorch

## Usage

This is a assembly of cuda extension including `Rotated_IoU`, `iou3d_nms`,  `pointnet2`, `roiaware_pool3d`, `roipoint_pool3d` and  `trilinear_interpolation`.

## Installation

a. Install and configure CUDA development kit properly.

b. Clone this repository.

```bash
git clone git@github.com:Uzukidd/cuda-extension-pytorch.git
```

c. Install pytorch.

d. Install this `cudaext` library by running the following command:

```bash
python setup.py develop
```

## Sources

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU)
- ### [pytorch-cppcuda-tutorial](https://github.com/kwea123/pytorch-cppcuda-tutorial)