import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder
from hamba.models.backbones.vmamba import Decorder_GCN_VSSM


def build_mano_head(cfg):
    mano_head_type = cfg.MODEL.MANO_HEAD.get('TYPE', 'hamba')
    if mano_head_type == 'mamba_gcn_decoder':
        return MANOMambaGCNDecoderHead(cfg)
    else:
        raise ValueError('Unknown MANO head type: {}'.format(mano_head_type))


class MANOMambaGCNDecoderHead(nn.Module):
    """ MANOMambaGCNDecoderHead
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.MANO_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)
        self.npose = npose
        
        self.dec_gcn_vssm = Decorder_GCN_VSSM(
                dims=cfg.MODEL.MANO_HEAD.DIMS,
                embed_dim=cfg.MODEL.MANO_HEAD.EMBED_DIM,
                depths=cfg.MODEL.MANO_HEAD.DEPTHS,
                ssm_d_state=cfg.MODEL.MANO_HEAD.SSM_D_STATE,
                ssm_dt_rank=cfg.MODEL.MANO_HEAD.SSM_DT_RANK,
                ssm_ratio=cfg.MODEL.MANO_HEAD.SSM_RATIO,
                ssm_conv=cfg.MODEL.MANO_HEAD.SSM_CONV,
                ssm_conv_bias=cfg.MODEL.MANO_HEAD.SSM_CONV_BIAS,
                forward_type=cfg.MODEL.MANO_HEAD.SSM_FORWARDTYPE,
                mlp_ratio=cfg.MODEL.MANO_HEAD.MLP_RATIO,
                downsample_version=cfg.MODEL.MANO_HEAD.DOWNSAMPLE,
                patchembed_version=cfg.MODEL.MANO_HEAD.PATCHEMBED,
                drop_path_rate=cfg.MODEL.MANO_HEAD.DROP_PATH_RATE,
                norm_layer=cfg.MODEL.MANO_HEAD.NORM_LAYER,
                joint_num=self.cfg.MODEL.JOINT_NUM,
                pretrained=None
            )


        dim=cfg.MODEL.MANO_HEAD.EMBED_DIM # for downsample the backbone dim
        if 'USE_JOINT2D_FEAT' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_JOINT2D_FEAT == True:
            dim = dim + 42
        if 'CONCAT_GLOBAL_MEAN_IN_END' in self.cfg.MODEL.keys() and self.cfg.MODEL.CONCAT_GLOBAL_MEAN_IN_END == True:
            dim = dim + cfg.MODEL.MANO_HEAD.EMBED_DIM
        if 'CONCAT_GRID_MEAN_IN_END' in self.cfg.MODEL.keys() and self.cfg.MODEL.CONCAT_GRID_MEAN_IN_END == True:
            dim = dim + cfg.MODEL.MANO_HEAD.EMBED_DIM

        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.MANO_HEAD.get('INIT_DECODER_XAVIER', False):
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(cfg.MANO.MEAN_PARAMS)
        init_hand_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        joint2d_feat = kwargs.get('joint2d_feat', None)
        global_feat_mean = kwargs.get('global_feat_mean', None)
        grid_feat_mean = kwargs.get('grid_feat_mean', None)
        
        init_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_hand_pose = init_hand_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_hand_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.cfg.MODEL.MANO_HEAD.get('IEF_ITERS', 1)):
            x = self.dec_gcn_vssm(x)[-1] 
            x = einops.rearrange(x, 'b c h w -> b (h w) c')
            token_out = x.mean(dim=1)

            if 'USE_JOINT2D_FEAT' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_JOINT2D_FEAT == True:
                joint2d_feat = joint2d_feat.reshape(batch_size, -1)
                token_out = torch.cat([token_out, joint2d_feat], dim=1)
            if 'CONCAT_GLOBAL_MEAN_IN_END' in self.cfg.MODEL.keys() and self.cfg.MODEL.CONCAT_GLOBAL_MEAN_IN_END == True:
                token_out = torch.cat([token_out, global_feat_mean], dim=1)
            if 'CONCAT_GRID_MEAN_IN_END' in self.cfg.MODEL.keys() and self.cfg.MODEL.CONCAT_GRID_MEAN_IN_END == True:
                token_out = torch.cat([token_out, grid_feat_mean], dim=1)

            # Readout from token_out
            pred_hand_pose = self.decpose(token_out) + pred_hand_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam
            pred_hand_pose_list.append(pred_hand_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_mano_params_list = {}
        pred_mano_params_list['hand_pose'] = torch.cat([joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_hand_pose_list], dim=0)
        pred_mano_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_mano_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(batch_size, self.cfg.MANO.NUM_HAND_JOINTS+1, 3, 3)

        pred_mano_params = {'global_orient': pred_hand_pose[:, [0]],
                            'hand_pose': pred_hand_pose[:, 1:],
                            'betas': pred_betas}
        return pred_mano_params, pred_cam, pred_mano_params_list
