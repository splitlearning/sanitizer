from torchvision import models
import torch.nn as nn
from  torch.nn.modules.loss import _Loss
import torch
import numpy as np

distance = nn.CrossEntropyLoss(reduction='sum')
mse = nn.MSELoss(reduction='sum')

class DistortionLoss(_Loss):
    def __init__(self, ):
        super(DistortionLoss, self).__init__()

    def forward(self, x_prime, x):
        return mse(x_prime, x)

class DistCorrelation(_Loss):
    def __init__(self, ):
        super(DistCorrelation, self).__init__()

    def get_centered(self, vec):
        return vec - vec.mean(dim=0, keepdim=True) - vec.mean(dim=1, keepdim=True) + vec.mean()

    def forward(self, z, data):
        z = z.reshape(z.shape[0], -1)
        data = data.reshape(data.shape[0], -1)
        a = torch.cdist(z, z)
        b = torch.cdist(data, data)
        a_centered = self.get_centered(a)
        b_centered = self.get_centered(b)
        dCOVab = torch.sqrt(torch.mean(a_centered * b_centered))
        var_aa = torch.sqrt(torch.mean(a_centered * a_centered))
        var_bb = torch.sqrt(torch.mean(b_centered * b_centered))

        dCORab = dCOVab / torch.sqrt(var_aa * var_bb)
        return dCORab


def ce_loss_fn(logits, labels):
    CE = distance(logits, labels)
    return CE

def reg_loss_fn(logits, labels):
    loss = mse(logits, labels)
    return loss



class ResNet50(nn.Module):
    """docstring for ResNet"""

    def __init__(self, config):
        super(ResNet50, self).__init__()
        self.logits = config["logits"]

        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
        return nn.functional.softmax(x)
