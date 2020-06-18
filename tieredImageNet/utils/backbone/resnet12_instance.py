import random

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

ceil = True
inp =True

class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5)

        self.convr = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bnr = nn.BatchNorm2d(planes, eps=2e-5)

        self.relu = nn.ReLU(inplace=inp)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)

    def forward(self, x, drop=False):
        identity = self.convr(x)
        identity = self.bnr(identity)

        if drop:
            out = self.relu(identity)
            out = self.maxpool(out)
            return out

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity

        out = self.relu(out)
        out = self.maxpool(out)

        return out


class ResNet12(nn.Module):
    def __init__(self, drop_ratio=0.1, with_drop=False):
        super(ResNet12, self).__init__()

        self.drop_layers = with_drop
        self.inplanes = 3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_ratio, inplace=inp)
        self.layer1 = ResNetBlock(self.inplanes, 64)
        self.inplanes = 64
        self.layer2 = ResNetBlock(self.inplanes, 128)
        self.inplanes = 128
        self.layer3 = ResNetBlock(self.inplanes, 256)
        self.inplanes = 256
        self.layer4 = ResNetBlock(self.inplanes, 512)
        self.inplanes = 512

        self.drop_pair = nn.Dropout(0.0, inplace=inp)
        self.layer1_rn = nn.Sequential(
                        nn.Conv2d(512,256,kernel_size=3,padding=0),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.fc1_rn = nn.Sequential(
                nn.Linear(256 * 2 * 2, 128),
                nn.BatchNorm1d(128, momentum=1, affine=True),
                nn.ReLU())
        self.fc2_rn = nn.Linear(128, 1)
        nn.init.xavier_uniform_(self.fc2_rn.weight)

        self.alpha = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.alpha, 0)
        self.beta = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.beta, 0)

        self.relu = nn.ReLU(inplace=inp)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.global_w = nn.Conv2d(in_channels=512, out_channels=351, kernel_size=1, stride=1)
        nn.init.xavier_uniform_(self.global_w.weight)

    def _make_layer(self, block, planes):
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.dropout(x)

        if self.drop_layers:
            x_f = self.layer4(x, drop=False)
            x_f = self.dropout(x_f)
            x_d = self.layer4(x, drop=True)
            x_d = self.dropout(x_d)
            key_list = [x_f, x_d]
            return key_list
        else:
            x_f = self.layer4(x, drop=False)
            x_f = self.dropout(x_f)
            key_list = [x_f]
            return key_list

    def instance_scale(self, set):

        sigma = self.layer1_rn(set)
        sigma = sigma.flatten(start_dim=1)
        sigma = self.fc1_rn(sigma)
        sigma = self.fc2_rn(sigma)
        sigma = torch.sigmoid(sigma)
        sigma = torch.exp(self.alpha) * sigma + torch.exp(self.beta)

        return sigma