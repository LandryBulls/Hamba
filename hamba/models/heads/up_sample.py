

import torch
import torch.nn as nn

class UpSampleOneDeconv(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, kernel_size=3, stride=2, padding=0):
        super(UpSampleOneDeconv, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_transpose(x)
        return x


class UpSample(nn.Module):
    def __init__(self, dim=1024, scale=4):
        super(UpSample, self).__init__()

        num_layers = int(scale / 2 - 1)  
        deconv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        up_layers = []
        for i in range(num_layers):
            up_layers.append(deconv)  

        self.deconv_layers = nn.Sequential(*up_layers) 
        self.deconv_last = nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.deconv_last(x)
        return x
