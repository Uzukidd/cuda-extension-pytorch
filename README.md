# cuda-extension-pytorch

## Usage

Collection  of cuda extension including `Rotated_IoU`, `iou3d_nms`,  `pointnet2`, `roiaware_pool3d`, `roipoint_pool3d` and  `trilinear_interpolation`.

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

## Operator list

- iou3d_nms (From [OpenPCDet](https://github.com/open-mmlab/OpenPCDet))
  - iou3d_nms_utils
    - boxes_bev_iou_cpu
    - boxes_iou_bev
    - boxes_iou3d_gpu
    - boxes_aligned_iou3d_gpu
    - nms_gpu
    - nms_normal_gpu
- pointnet2
  - pointnet2_batch (From [OpenPCDet](https://github.com/open-mmlab/OpenPCDet))
    - pointnet2_utils
      - FarthestPointSampling
      - GatherOperation
      - ThreeNN
      - ThreeInterpolate
      - GroupingOperation
      - BallQuery
      - QueryAndGroup
      - GroupAll
    - pointnet2_modules
      - PointnetSAModuleMSG
      - PointnetSAModule
      - PointnetFPModule
  - pointnet2_stack (From [OpenPCDet](https://github.com/open-mmlab/OpenPCDet))
    - pointnet2_utils
      - ...
    - pointnet2_modules
      - ...
    - voxel_pool_modules
      - NeighborVoxelSAModuleMSG
    - voxel_query_utils
      - VoxelQuery
      - VoxelQueryAndGrouping
- roiaware_pool3d (From [OpenPCDet](https://github.com/open-mmlab/OpenPCDet))
  - roiaware_pool3d_utils
    - points_in_boxes_cpu
    - points_in_boxes_gpu
    - RoIAwarePool3d
    - RoIAwarePool3dFunction
- roipoint_pool3d (From [OpenPCDet](https://github.com/open-mmlab/OpenPCDet))
  - roipoint_pool3d_utils
    - RoIPointPool3d
    - RoIPointPool3dFunction
- Rotated_IoU (From [Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU))
  - box_intersection_2d
    - box_intersection_th
    - box1_in_box2
    - box_in_box_th
    - build_vertices
    - sort_indices
    - calculate_area
    - oriented_box_intersection_2d
  - min_enclosing_box
    - generate_table
    - gather_lines_points
    - point_line_distance_range
    - point_line_projection_range
    - smallest_bounding_box
  - oriented_iou_loss
    - box2corners_th
    - cal_iou
    - cal_diou
    - cal_giou
    - cal_iou_3d
    - cal_giou_3d
    - cal_diou_3d
    - enclosing_box
    - enclosing_box_aligned
    - enclosing_box_pca
    - eigenvector_22
    - assign_target_3d
  - utiles
    - line_seg_intersection
    - box2corners
    - box_intersection
    - point_in_box
    - box_in_box
    - intersection_poly
    - compare_vertices
    - vertices2area
    - box_intersection_area
- trilinear_interpolate
  - trilinear_interpolate_utils
    - Trilinear_interpolation_cuda
    - trilinear_interpolation_cpu

## Sources

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU)
- ### [pytorch-cppcuda-tutorial](https://github.com/kwea123/pytorch-cppcuda-tutorial)
