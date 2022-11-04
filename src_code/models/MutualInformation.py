# Taken from https://github.com/charleslipku/TIPRDC/blob/main/Image/SegmentVGG16.py

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MutlInfo(nn.Module):
    def __init__(self, num_classes=4):
        super(MutlInfo, self).__init__()
        self.downscale = nn.Conv2d(128, 64, 1)
        self.convnet_1 = nn.Sequential(
            nn.Conv2d(3, 64//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64//2, 64//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64//2, 128//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(128//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128//2, 128//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(128//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convnet_2 = nn.Sequential(
            nn.Conv2d(128//2, 256//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(256//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256//2, 256//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(256//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256//2, 256//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(256//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256//2, 512//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(512//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//2, 512//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(512//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//2, 512//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(512//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512//2, 512//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(512//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//2, 512//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(512//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//2, 512//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(512//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fcnet = nn.Sequential(
            nn.Linear(8192 + num_classes, 4096//2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096//2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 1)
        )

        # initialize weight

    def forward(self, x, z, u):
        out1 = self.convnet_1(x)
        out1 = self.convnet_2(out1)
        out1 = out1.view(out1.size(0), -1)

        z = self.downscale(z)
        out2 = self.convnet_2(z)
        out2 = out2.view(out2.size(0), -1)

        out = torch.cat((out1, out2, u), dim=1)
        out = self.fcnet(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def info_loss(MI, x, z, u, x_prime):
    Ej = -F.softplus(-MI(x, z, u)).mean()
    Em = F.softplus(MI(x_prime, z, u)).mean()
    return Ej - Em
