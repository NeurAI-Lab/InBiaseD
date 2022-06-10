import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp(nn.Module):
    def __init__(self, num_classes=10, cifar_resnet=False, inplanes=28, dim=512):
        super(mlp, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(inplanes*inplanes*3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(nn.Linear(dim, dim),
                                     nn.ReLU())
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        self.feat = []

        out = torch.flatten(x,1)
        out = self.layers(out)
        self.feat.append(out)
        out = self.layer2(out)
        self.feat.append(out)
        out = self.fc(out)

        return out

