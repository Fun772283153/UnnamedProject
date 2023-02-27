import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import math

from ..builder import ROTATED_HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.utils import reduce_mean
from mmcv.cnn import Conv2d
from ..utils.rotated_dab_transformer import MLP


@ROTATED_HEADS.register_module()
class RotatedDabDetrHead(DETRHead):
    def __init__(
        self,
        angle_version: str,
        bbox_embed_diff_each_layer: bool = False,
        query_dim: int = 5,
        random_refpoints_xy: bool = False,
        aux_loss: bool = False,
        iter_update: bool = True,
        **kwargs
    ):
        assert angle_version in ['oc', 'le90', 'le135']
        self.angle_version = angle_version
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.query_dim = query_dim
        self.random_refpoints_xy = random_refpoints_xy
        self.aux_loss = aux_loss
        self.iter_update = iter_update
        super(RotatedDabDetrHead, self).__init__(**kwargs)

    
    def _init_layers(self):
        self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        self.class_embed = nn.Linear(self.embed_dims, self.cls_out_channels)
        if self.bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(self.embed_dims, self.embed_dims, self.query_dim, 3)])
        else:
            self.bbox_embed = MLP(self.embed_dims, self.embed_dims, self.query_dim, 3)
        self.refpoint_embed = nn.Embedding(self.num_query, self.query_dim)
        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed

    def init_weights(self):
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
        if self.bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value

    
    def forward_single(self, x, img_metas):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0
        
        x = self.input_proj(x)
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]
        ).to(torch.bool).squeeze(1)

        pos_embed = self.positional_encoding(masks)
        embed_weight = self.refpoint_embed.weight
        hs, reference = self.transformer(x, masks, embed_weight, pos_embed)

        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(reference)
            tmp = self.bbox_embed(hs)
            tmp[..., :self.query_dim] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        return outputs_class, outputs_coord
    
    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, gt_bboxes_list,
            gt_labels_list, img_metas, gt_bboxes_ignore_list,
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h, np.pi / 2]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        bbox_preds = bbox_preds.reshape(-1, 5)
        bboxes = bbox_preds * factors
        bboxes_gt = bbox_targets * factors

        loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou
    

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        gt_bboxes,
        gt_labels,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        num_bboxes = bbox_pred.size(0)

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h, np.pi / 2]).unsqueeze(0)

        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        bbox_targets[pos_inds] = pos_gt_bboxes_normalized
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)
    

    def _get_bboxes_single(
        self,
        cls_score,
        bbox_pred,
        img_shape,
        scale_factor,
        rescale=False,
    ):
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        img_h, img_w, _ = img_shape
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h, np.pi / 2]).unsqueeze(0)
        bbox_pred = bbox_pred * factor
        if rescale:
            bbox_pred[:, :4] /= bbox_pred[:, :4].new_tensor(scale_factor)
        det_bboxes = torch.cat((bbox_pred, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels


        
