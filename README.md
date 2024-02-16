# cuda-extension-pytorch

## Usage

This is a assembly of cuda extension including `Rotated_IoU`, `iou3d_nms`,  `pointnet2`, `roiaware_pool3d` and `roipoint_pool3d`.

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

- https://github.com/open-mmlab/OpenPCDet
- https://github.com/lilanxiao/Rotated_IoU