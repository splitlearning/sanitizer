# Taken from https://github.com/charleslipku/TIPRDC/blob/main/Image/SegmentVGG16.py
from torchvision import models
import torch.nn as nn
import math

def reconstruction_loss(img1, img2):
    return nn.L1Loss()(img1, img2)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.convnet(x)
        return out


class Classifier(nn.Module):
    def __init__(self, attributes, num_classes=2, split_layer=6):
        super(Classifier, self).__init__()
        model = list(models.resnet34(pretrained=True).children())
        fc_features = model[-1].in_features
        model[-1] = nn.Sequential(nn.Flatten(),
                                    nn.Linear(fc_features, num_classes))
        model = nn.ModuleList(model)[split_layer:]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        y = nn.functional.softmax(self.model(x))
        return y

