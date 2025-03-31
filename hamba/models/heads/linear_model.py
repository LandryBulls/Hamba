from __future__ import absolute_import

import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, linear_size=1024, num_stage=2, p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size = input_size  # 16 * 2
        # 3d joints
        self.output_size = output_size  # 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        if self.num_stage > 0:
            self.linear_stages = []
            for l in range(num_stage):
                self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
            self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y

class TinyLinear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(TinyLinear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.relu(y)
        y = self.dropout(y)

        return y


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    
class Linear_LN_AVGPOOL(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, depth_num=1):
        super(Linear_LN_AVGPOOL, self).__init__()
        '''
        ln=nn.LayerNorm,
        ln2d=LayerNorm2d,
        bn=nn.BatchNorm2d,
        '''
        self.depth_num = depth_num
        one_linear = nn.Sequential(OrderedDict(
            norm=nn.BatchNorm2d(input_dim), # B,H,W,C
            head=nn.Linear(input_dim, output_dim),
        ))

        self.linear_stages = []
        for l in range(depth_num - 1):
            self.linear_stages.append(one_linear)
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.last_layer = nn.Sequential(OrderedDict(
            norm=nn.LayerNorm(input_dim), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
        ))

        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        for i in range(self.depth_num - 1):
            x = self.linear_stages[i](x)
        y = self.last_layer(x)

        return y
    