
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GridSampleFeature(nn.Module):
    def __init__(self, in_dim=1920, out_dim=1920):
        super(GridSampleFeature, self).__init__()
        self.filters = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, 1),
        )

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_feature, joint_uv, mode: str = "bilinear",):
        sampled_mesh_feat = F.grid_sample(img_feature, joint_uv.unsqueeze(1).detach(), mode=mode).squeeze(-2)
        sampled_mesh_feat = self.filters(sampled_mesh_feat)
        
        return sampled_mesh_feat

class GridSampleFeatureFC(nn.Module):
    def __init__(self, in_dim=1920, out_dim=1920):
        super(GridSampleFeature, self).__init__()

        self.out_dim = out_dim
        joint_num = 21
        self.filters = nn.Sequential(
            nn.Linear(in_dim * joint_num, out_dim * joint_num),
            nn.BatchNorm1d(out_dim * joint_num),
            nn.ReLU(inplace=True)
        )
        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_feature, joint_uv):
        sampled_mesh_feat = F.grid_sample(img_feature, joint_uv.unsqueeze(1).detach()).squeeze(-2)
        B, C, H = sampled_mesh_feat.shape
        sampled_mesh_feat = sampled_mesh_feat.view(B, -1) # (B, C)
        sampled_mesh_feat = self.filters(sampled_mesh_feat)
        sampled_mesh_feat = sampled_mesh_feat.reshape(B, self.out_dim, H)
        
        return sampled_mesh_feat