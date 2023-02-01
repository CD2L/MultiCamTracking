import torch
from torch import nn
from torchvision.models import resnet50

class SiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50()

    def forward(self, anchor, pos, neg):
        in_anc = self.encoder(anchor)
        in_pos = self.encoder(pos)
        in_neg = self.encoder(neg)

        return in_anc, in_pos, in_neg