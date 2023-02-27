from mmcv.utils import build_from_cfg
from mmdet.models.utils.builder import TRANSFORMER

ROTATED_TRANSFORMER = TRANSFORMER

def build_transformer(cfg, default_args=None):
    return build_from_cfg(cfg, ROTATED_TRANSFORMER, default_args)