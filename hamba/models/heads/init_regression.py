import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat

from .linear_model import LinearModel, init_weights, TinyLinear, Linear_LN_AVGPOOL
from hamba.models.backbones.vmamba import Decorder_VSSM


class InitRegression(nn.Module):
    """ Initial regressor for obtaining joint_2d and joint3d
    """

    def __init__(self, cfg, dim=1024):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.MANO_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)
       
        if 'USE_MAMBA_BLOCKS' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_MAMBA_BLOCKS == True:
            self.dec_vssm = Decorder_VSSM(
                dims=cfg.MODEL.DECODER_MAMBA.DIMS,
                embed_dim=cfg.MODEL.DECODER_MAMBA.EMBED_DIM,
                depths=cfg.MODEL.DECODER_MAMBA.DEPTHS,
                ssm_d_state=cfg.MODEL.DECODER_MAMBA.SSM_D_STATE,
                ssm_dt_rank=cfg.MODEL.DECODER_MAMBA.SSM_DT_RANK,
                ssm_ratio=cfg.MODEL.DECODER_MAMBA.SSM_RATIO,
                ssm_conv=cfg.MODEL.DECODER_MAMBA.SSM_CONV,
                ssm_conv_bias=cfg.MODEL.DECODER_MAMBA.SSM_CONV_BIAS,
                forward_type=cfg.MODEL.DECODER_MAMBA.SSM_FORWARDTYPE,
                mlp_ratio=cfg.MODEL.DECODER_MAMBA.MLP_RATIO,
                downsample_version=cfg.MODEL.DECODER_MAMBA.DOWNSAMPLE,
                patchembed_version=cfg.MODEL.DECODER_MAMBA.PATCHEMBED,
                drop_path_rate=cfg.MODEL.DECODER_MAMBA.DROP_PATH_RATE,
                norm_layer=cfg.MODEL.DECODER_MAMBA.NORM_LAYER,
                pretrained=None
            )

        ## from https://github.com/garyzhao/SemGCN/blob/87fe4a5e43d27a361376df47782c521566601505/models/linear_model.py#L60
        elif 'DEEP_LINEAR_NORM_RELU_DROP' in self.cfg.MODEL.keys() and self.cfg.MODEL.DEEP_LINEAR_NORM_RELU_DROP == True:
            if 'DEEP_LINEAR_NUM' in self.cfg.MODEL.keys():
                num_stage = self.cfg.MODEL.DEEP_LINEAR_NUM
            else:
                num_stage = 2

            if 'IS_TINY_LINEAR' in self.cfg.MODEL.keys() and self.cfg.MODEL.IS_TINY_LINEAR == True:
                self.linearModel = TinyLinear(linear_size=dim)
            else:
                self.linearModel = LinearModel(dim, dim, linear_size=1024, num_stage=num_stage, p_dropout=0.5)
            self.linearModel.apply(init_weights)

        elif 'USE_LINEAR_LN_AVGPOOL' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_LINEAR_LN_AVGPOOL == True:
            depth_num = self.cfg.MODEL.DEEP_LINEAR_NUM
            self.linearModel = Linear_LN_AVGPOOL(input_dim=dim, output_dim=dim, depth_num=depth_num)
        else:
            print("None Init_regression!")
        
        self.decpose = nn.Linear(dim, npose) 
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.MANO_HEAD.get('INIT_DECODER_XAVIER', False): # default is None
            # True by default in MLP. False by default in Transformer
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

        if 'DEEP_LINEAR_NORM_RELU_DROP' in self.cfg.MODEL.keys() and self.cfg.MODEL.DEEP_LINEAR_NORM_RELU_DROP == True \
            or 'USE_LINEAR_LN_AVGPOOL' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_LINEAR_LN_AVGPOOL == True:
            x = self.linearModel(x)

        elif 'USE_MAMBA_BLOCKS' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_MAMBA_BLOCKS == True:
            x = self.dec_vssm(x)[-1]
            x = einops.rearrange(x, 'b c h w -> b (h w) c')
            x = x.mean(dim=1)

        init_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # TODO: Convert init_hand_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_hand_pose = init_hand_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_hand_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.cfg.MODEL.MANO_HEAD.get('IEF_ITERS', 1)):
            pred_hand_pose = self.decpose(x) + pred_hand_pose
            pred_betas = self.decshape(x) + pred_betas
            pred_cam = self.deccam(x) + pred_cam
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
