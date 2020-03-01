import torch.nn as nn


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )

class ConvNet(nn.Module):

    def __init__(self, with_drop=False):
        super().__init__()

        self.drop_layer = with_drop

        self.hidden = 64
        self.layer1 = conv_block(3,  self.hidden)
        self.layer2 = conv_block(self.hidden, int(1.5 * self.hidden))
        self.layer3 = conv_block(int(1.5 * self.hidden), 2 * self.hidden)
        self.layer4 = conv_block(2 * self.hidden,   4 * self.hidden)

        self.weight = nn.Linear(4 * self.hidden, 64)
        nn.init.xavier_uniform_(self.weight.weight)

        self.conv1_ls = nn.Conv2d(in_channels=4 * self.hidden, out_channels=1, kernel_size=3)
        self.bn1_ls = nn.BatchNorm2d(1, eps=2e-5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1_ls = nn.Linear(16, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        if self.drop_layer:
            return [x4, x3]
        else:
            return [x4]