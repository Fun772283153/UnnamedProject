import torch
import torch.nn as nn
import numpy as np

from scipy.optimize import linear_sum_assignment
from ..builder import ROTATED_BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.assigners.assign_result import AssignResult


@ROTATED_BBOX_ASSIGNERS.register_module()
class RotatedHuangarianAssigner(BaseAssigner):
    def __init__(
        self,
        cls_cost=dict(type='ClassificationCost', weight=1.0),
        reg_cost=dict(type='BBoxL1Cost', weigh=1.0),
        iou_cost=dict(type='IoUCost', iou_mode='iou', weight=1.0)
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def assign(
        self,
        bbox_pred,
        cls_pred,
        gt_bboxes,
        gt_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7
    ):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
    
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h, np.pi / 2]).unsqueeze(0)

        cls_cost = self.cls_cost(cls_pred, gt_labels)
        normalize_gt_rbboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_rbboxes)
        iou_cost = self.iou_cost(bbox_pred, normalize_gt_rbboxes)

        cost = cls_cost + reg_cost + iou_cost

        cost = cost.detach().cpu()

        cost_matrix = np.asarray(cost)
        contain_nan = (True in np.isnan(cost_matrix))
        if contain_nan:
            print(img_meta['file_name'])
        
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

