from mmcv.utils import build_from_cfg
from mmdet.core.bbox.match_costs.builder import MATCH_COST

ROTATED_MATCH_COST = MATCH_COST

def build_match_cost(cfg, default_args=None):
    return build_from_cfg(cfg, ROTATED_MATCH_COST, default_args)
