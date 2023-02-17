import torch
from torch import nn
from torchvision.models import resnet50, resnet101
import numpy as np
from torchreid.reid.utils import FeatureExtractor

class BaseModel(nn.Module):
    def __init__(self, parallele = False, partial_freeze = False):
        super().__init__()
        self.encoder = None
        pass

    def preprocessing(self, x: np.ndarray):
        if len(x.shape) == 3:
            tensor = torch.from_numpy(x).permute(2,0,1)
        else: 
            tensor = torch.from_numpy(x).permute(0,3,1,2)
        return tensor/255

    def forward(self, x):
        return self.encoder(x)


class SiameseModel(BaseModel):
    def __init__(self, parallele = False, partial_freeze = False):
        super().__init__()
        self.encoder = resnet50(weights ='DEFAULT')            

        ct = 0
        if partial_freeze:
            for child in self.encoder.children():
                ct += 1
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False

        if parallele:
            self.encoder = nn.DataParallel(self.encoder)

class SMResNet101(BaseModel):
    def __init__(self, parallele = False, partial_freeze = False):
        super().__init__()
        self.encoder = resnet101(weights ='DEFAULT')            

        ct = 0
        if partial_freeze:
            for child in self.encoder.children():
                ct += 1
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False

        if parallele:
            self.encoder = nn.DataParallel(self.encoder)

class torchereidModels():
    def __init__(self, name='osnet_ain_x1_0', weights = None) -> None:
        self.encoder = FeatureExtractor(
            model_name=name,
            model_path=weights,
            device='cuda'
        )

    def __call__(self, x):
        return self.encoder(x)

