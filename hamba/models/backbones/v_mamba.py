
import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from hamba.models.backbones.vmamba import Backbone_VSSM


def v_mamba(cfg):
    vmamba_backbone = Backbone_VSSM(
        dims=cfg.MODEL.BACKBONE.DIMS,
        embed_dim=cfg.MODEL.BACKBONE.EMBED_DIM,
        depths=cfg.MODEL.BACKBONE.DEPTHS,
        ssm_d_state=cfg.MODEL.BACKBONE.SSM_D_STATE,
        ssm_dt_rank=cfg.MODEL.BACKBONE.SSM_DT_RANK,
        ssm_ratio=cfg.MODEL.BACKBONE.SSM_RATIO,
        ssm_conv=cfg.MODEL.BACKBONE.SSM_CONV,
        ssm_conv_bias=cfg.MODEL.BACKBONE.SSM_CONV_BIAS,
        forward_type=cfg.MODEL.BACKBONE.SSM_FORWARDTYPE,
        mlp_ratio=cfg.MODEL.BACKBONE.MLP_RATIO,
        downsample_version=cfg.MODEL.BACKBONE.DOWNSAMPLE,
        patchembed_version=cfg.MODEL.BACKBONE.PATCHEMBED,
        drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE,
        norm_layer=cfg.MODEL.BACKBONE.NORM_LAYER,
        pretrained=None
    )

    return vmamba_backbone
