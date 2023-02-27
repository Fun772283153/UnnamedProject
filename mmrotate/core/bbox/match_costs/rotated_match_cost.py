import math
import torch

from .builder import build_match_cost, ROTATED_MATCH_COST
from mmcv.ops import box_iou_rotated

@ROTATED_MATCH_COST.register_module()
class RBBoxL1Cost:
    def __init__(
        self,
        weight=1,
        box_format='xyxya'
    ) -> None:
        self.weight = weight
        assert box_format in ['xyxya', 'xywha']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
    

@ROTATED_MATCH_COST.register_module()
class RotatedIoUCost:
    def __init__(
        self,
        iou_mode='giou',
        weight=1.0,
    ) -> None:
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        overlaps = box_iou_rotated(bboxes, gt_bboxes)
        iou_cost = -overlaps
        return iou_cost * self.weight
    
