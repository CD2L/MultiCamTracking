import torch
from torch import nn
from torchvision.models import resnet50
import numpy as np

class SiameseModel(nn.Module):
    def __init__(self, parallele = False):
        super().__init__()
        self.encoder = resnet50()
        if parallele:
            self.encoder = nn.DataParallel(self.encoder)

    def preprocessing(self, x: np.ndarray):
        if len(x.shape) == 3:
            tensor = torch.from_numpy(x).permute(2,0,1)
        else: 
            tensor = torch.from_numpy(x).permute(0,3,1,2)
        return tensor/255

    def forward(self, x):
        return self.encoder(x)