
from .vit import vit
from .v_mamba import v_mamba
# from .fastvit import fastvit_ma36
import timm
from timm.models.layers import DropPath, Mlp

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
