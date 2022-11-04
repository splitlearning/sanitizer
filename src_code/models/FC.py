import torch.nn as nn

class FC(nn.Module):
    def __init__(self, config):
        super(FC, self).__init__()
        layer1 = nn.Linear(config["inp_dim"], 128)
        act1 = nn.ReLU()
        layer2 = nn.Linear(128, config["logits"])
        self.model = nn.Sequential(layer1, act1, layer2)

    def forward(self, x):
        return nn.functional.softmax(self.model(x), dim=1)
