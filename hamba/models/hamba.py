import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_mano_head
from .discriminator import Discriminator
from . import MANO
import torch.nn.functional as F
import einops

from .heads.grid_sample_feature import GridSampleFeature
from .heads.up_sample import UpSample, UpSampleOneDeconv
from .heads.init_regression import InitRegression
import torch.nn as nn
import numpy as np

log = get_pylogger(__name__)

class HAMBA(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup HAMBA model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            checkpoint_path = cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            if self.cfg.MODEL.BACKBONE.TYPE == 'vit':
                if 'vitpose_backbone.pth' in checkpoint_path:
                    self.backbone.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
                else:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    backbone_state_dict = {}
                    for name, param in checkpoint['state_dict'].items():
                        if name.startswith('backbone.'):
                            backbone_state_dict[name.replace('backbone.', '')] = param
                    self.backbone.load_state_dict(backbone_state_dict)
            log.info(f'Loaded backbone weights from {checkpoint_path}, done!')

        # Create MANO head
        self.mano_head = build_mano_head(cfg)

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        e_dim = 1024
        if 'FEATURE_FUSION_TYPE' in self.cfg.MODEL.keys():
            if self.cfg.MODEL.FEATURE_FUSION_TYPE == 'layer4':
                if self.cfg.MODEL.BACKBONE.TYPE == 'vit':
                    e_dim = 1280

        if 'USE_DOWNSAMPLE' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_DOWNSAMPLE == True:
            context_dim = 512
            self.downsample = nn.Sequential(
                nn.Conv2d(e_dim, context_dim, 1),
                nn.BatchNorm2d(context_dim),
                nn.ReLU(),
                nn.Conv2d(context_dim, context_dim, 1),
            )
        else:
            context_dim = e_dim

        if 'USE_INIT_REGRESSION' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_INIT_REGRESSION == True:
            self.initRegression = InitRegression(cfg, dim=context_dim)

        if 'GRIDSAMPLE' in self.cfg.MODEL.keys() and self.cfg.MODEL.GRIDSAMPLE == True:
            self.gridSampleFeature = GridSampleFeature(in_dim=context_dim, out_dim=context_dim)

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def get_parameters(self):
        all_params = list(self.mano_head.parameters())
        all_params += list(self.backbone.parameters())
        if 'GRIDSAMPLE' in self.cfg.MODEL.keys() and self.cfg.MODEL.GRIDSAMPLE == True:
            all_params += list(self.gridSampleFeature.parameters())
        if 'FEATURE_FUSION_TYPE' in self.cfg.MODEL.keys() and self.cfg.MODEL.FEATURE_FUSION_TYPE == 'layer4_up28x28':
            all_params += list(self.upsample.parameters())
        if 'USE_INIT_REGRESSION' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_INIT_REGRESSION == True:
            all_params += list(self.initRegression.parameters())
        if 'USE_DOWNSAMPLE' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_DOWNSAMPLE == True:
            all_params += list(self.downsample.parameters())

        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        if self.cfg.MODEL.BACKBONE.TYPE == 'vit':
            conditioning_feats = self.backbone(x[:,:,:,32:-32]) # 256 x 196
            if 'USE_DOWNSAMPLE' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_DOWNSAMPLE == True:
                conditioning_feats = self.downsample(conditioning_feats) 

        _, token_dim, token_h, token_w = conditioning_feats.shape
        conditioning_feats_all = conditioning_feats

        ######### Initial Regressor ###########
        conditioning_feats_mean = conditioning_feats.mean(dim=(-2, -1), keepdim=False)  
        if 'USE_INIT_REGRESSION' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_INIT_REGRESSION == True:
            if 'USE_MAMBA_BLOCKS' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_MAMBA_BLOCKS == True:
                pred_mano_params, pred_cam, _ = self.initRegression(conditioning_feats)
            else:
                pred_mano_params, pred_cam, _ = self.initRegression(conditioning_feats_mean)
            output_init_regression = self.build_output_dict(pred_cam, pred_mano_params)

        if 'ONLY_TRAIN_USE_INIT_REGRESSION' in self.cfg.MODEL.keys() and self.cfg.MODEL.ONLY_TRAIN_USE_INIT_REGRESSION == True:
            # disable GAN loss
            if not isinstance(self.cfg, CfgNode):
                self.cfg.LOSS_WEIGHTS.ADVERSARIAL = 0 
            else:
                self.cfg.defrost()
                self.cfg.LOSS_WEIGHTS.ADVERSARIAL = 0 
                self.cfg.freeze()

            ret_output = {
                'output_init_regression': output_init_regression
            }
            return ret_output
        ##########################################

        if 'GRIDSAMPLE' in self.cfg.MODEL.keys() and self.cfg.MODEL.GRIDSAMPLE == True:
            if 'USE_INIT_REGRESSION' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_INIT_REGRESSION == True:
                joint_2d_norm = output_init_regression['pred_keypoints_2d']
            else:
                joint_2d_norm = batch['keypoints_2d'] 
            joint_2d_norm_minus11 = joint_2d_norm[:, :, :2] * 2 

            if 'CLIP_JOINT2D' in self.cfg.MODEL.keys() and self.cfg.MODEL.CLIP_JOINT2D == True:
                pad_value = (token_h - token_w) * 0.5 / token_h
                clip_min = -1 + pad_value
                clip_max = 1 - pad_value
                joint_2d_norm_x = joint_2d_norm_minus11[:, :, 0]
                joint_2d_norm_x_clipped = torch.clamp(joint_2d_norm_x, min=clip_min, max=clip_max)
                joint_2d_norm_x_new = joint_2d_norm_x_clipped / (1 - pad_value)
                joint_2d_norm_minus11[:, :, 0] = joint_2d_norm_x_new

            mode = "bilinear"
            if 'GRID_SAMPLE_MODE' in self.cfg.MODEL.keys():
                mode = self.cfg.MODEL.GRID_SAMPLE_MODE

            if 'GRIDSAMPLE_ORIGIANL' in self.cfg.MODEL.keys() and self.cfg.MODEL.GRIDSAMPLE_ORIGIANL == True:
                grid_sample_feature = F.grid_sample(conditioning_feats, joint_2d_norm_minus11.unsqueeze(1).detach(), mode=mode).squeeze(-2)
            else:
                grid_sample_feature = self.gridSampleFeature(conditioning_feats, joint_2d_norm_minus11, mode=mode)
            conditioning_feats_all = grid_sample_feature

        else:
            conditioning_feats_all = conditioning_feats_mean.unsqueeze(-1).unsqueeze(-1)

        if 'CONCAT_GLOBAL_MEAN' in self.cfg.MODEL.keys() and self.cfg.MODEL.CONCAT_GLOBAL_MEAN == True:
            conditioning_feats_all = torch.cat([grid_sample_feature, conditioning_feats_mean.unsqueeze(-1)], dim=-1)     # 4, 22, 512
            conditioning_feats_all = conditioning_feats_all.unsqueeze(-1)

        if 'USE_JOINT2D_FEAT' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_JOINT2D_FEAT == True:
            joint2d_feat = joint_2d_norm
        else:
            joint2d_feat = None
        
        if 'CONCAT_GLOBAL_MEAN_IN_END' in self.cfg.MODEL.keys() and self.cfg.MODEL.CONCAT_GLOBAL_MEAN_IN_END == True:
            global_feat_mean = conditioning_feats_mean
        else:
            global_feat_mean = None

        if 'CONCAT_GRID_MEAN_IN_END' in self.cfg.MODEL.keys() and self.cfg.MODEL.CONCAT_GRID_MEAN_IN_END == True:
            grid_feat_mean = grid_sample_feature.mean(dim=-1, keepdim=False)  
        else:
            grid_feat_mean = None 

        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats_all, joint2d_feat=joint2d_feat, global_feat_mean=global_feat_mean, grid_feat_mean=grid_feat_mean)

        output = self.build_output_dict(pred_cam, pred_mano_params)

        # combine output and output_init_regression
        if 'USE_INIT_REGRESSION' in self.cfg.MODEL.keys() and self.cfg.MODEL.USE_INIT_REGRESSION == True:
            output['output_init_regression'] = output_init_regression

        return output

    def build_output_dict(self, pred_cam, pred_mano_params):
        batch_size = pred_cam.shape[0]

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}

        # Compute camera translation
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)

        return output

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        ret_output = self.forward_step(batch, train=False)

        if 'ONLY_TRAIN_USE_INIT_REGRESSION' in self.cfg.MODEL.keys() and self.cfg.MODEL.ONLY_TRAIN_USE_INIT_REGRESSION == True:
            return ret_output['output_init_regression'] # for the init_regression stage
        else:
            return ret_output

    def training_step_discriminator(self, batch: Dict,
                                    hand_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            hand_pose (torch.Tensor): Regressed hand pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = hand_pose.shape[0]
        gt_hand_pose = batch['hand_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_hand_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(hand_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)

        output, loss, pred_mano_params = self.compute_and_combine_losses(batch, output, train=True)

        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)

        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()

        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        output = self.forward_step(batch, train=False)

        output, loss, pred_mano_params = self.compute_and_combine_losses(batch, output, train=False)

        output['loss'] = loss

        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output
