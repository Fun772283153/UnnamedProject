import warnings

import torch
import torch.nn as nn
import numpy as np

from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector
from .utils import FeatureRefineModule

@ROTATED_DETECTORS.register_module()
class RotatedDETR(RotatedSingleStageDetector):
    def __init__(
        self,
        backbone,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(RotatedDETR, self).__init__(backbone, None, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)

    
    def rbox2result(self, bboxes, labels, num_classes):
        """ Covert detection results to a list of numpy arrays.
        Args:
            bboxes (torch.Tensor): shape(n, 5)
            labels (torch.Tensor): shape(n, )
            num_classes (int): class number
        Returns:
            list(ndarray): bbox results of each class        
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 9), dtype=np.float32) for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            
            return [bboxes[labels == i, :] for i in range(num_classes)]
        
    def forward_dummy(self, img):
        warnings.warn('Warning! MultiheadAttention in DETR does not'
                      'support flops computation! Do not use the results'
                      'in your papers!')
        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)
            ) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs
    
    def onnx_export(self, img, img_metas):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas)
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas)
        return det_bboxes, det_labels

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        cost_matrix = np.asarray(x[0].detach().cpu())
        contain_nan = (True in np.isnan(cost_matrix))
        if contain_nan:
            print('Find!!!')
            for i in range(len(img_metas)):
                print("The image is", img_metas[i]['file_name'])
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses
    
    def imshow_gpu_tensor(self, tensor):
        from PIL import Image
        from torchvision import transforms
        device = tensor[0].device
        mean = torch.tensor([123.675, 116.28, 103.53]).to(device)
        std = torch.tensor([58.395, 57.12, 57.375]).to(device)
        tensor = (tensor[0].squeeze() * std[:, None, None]) + mean[:, None, None]
        tensor = tensor[0:1]
        if len(tensor.shape) == 4:
            image = tensor.permute(0, 2, 3, 1).cpu().clone().numpy()
        else:
            image = tensor.permute(1, 2, 0).cpu().clone().numpy()
        image = image.astype(np.uint8).squeeze()
        image = transforms.ToPILImage()(image)
        image = image.resize((256, 256), Image.ANTIALIAS)
        image.show(image)

    def simple_test(self, img, img_metas, rescale=False):
        cfg = self.test_cfg

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test_bboxes(feat, img_metas, rescale=rescale)
        bbox_results = [
            self.rbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    
