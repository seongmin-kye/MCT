import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


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

    def forward(self, x):
        identity = self.convr(x)
        identity = self.bnr(identity)

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
        self.layer1 = self._make_layer(ResNetBlock, 64)
        self.layer2 = self._make_layer(ResNetBlock, 128)
        self.layer3 = self._make_layer(ResNetBlock, 256)
        self.layer4 = self._make_layer(ResNetBlock, 512)

        # global weight
        self.weight = nn.Linear(512, 64)
        nn.init.xavier_uniform_(self.weight.weight)

        # length scale parameters
        self.conv1_ls = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3)
        self.bn1_ls = nn.BatchNorm2d(1, eps=2e-5)
        self.fc1_ls = nn.Linear(16, 1)

        self.relu = nn.ReLU(inplace=inp)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x3 = self.dropout(x)

        x = self.layer4(x3)
        x4 = self.dropout(x)

        if self.drop_layers:
            return [x4, x3]
        else:
            return [x4]