import torch
from torch import nn
from torchvision.models import resnet50, resnet101
import numpy as np
from PIL import Image
from torchreid.reid.utils import FeatureExtractor
import torchreid
from torchreid import metrics
from torchreid.reid.data.transforms import build_transforms


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

class REID:
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.model = torchreid.models.build_model(
            name='resnet50',
            num_classes=1,  # human
            loss='softmax',
            pretrained=True,
            use_gpu=self.use_gpu
        )

        torchreid.utils.load_pretrained_weights(self.model, 'siamese_model/checkpoints-saved/model.pth')

        if self.use_gpu:
            self.model = self.model.cuda()
            _, self.transform_te = build_transforms(
                height=256, width=128,
                random_erase=False,
                color_jitter=False,
                color_aug=False
            )
        self.dist_metric = 'euclidean'
        self.model.eval()

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _features(self, imgs):
        f = []
        for img in imgs:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = self.transform_te(img)
            img = torch.unsqueeze(img, 0)
            if self.use_gpu:
                img = img.cuda()
            features = self._extract_features(img)
            features = features.data.cpu()  # tensor shape=1x2048
            f.append(features)
        f = torch.cat(f, 0)
        return f

    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        return distmat.numpy()
