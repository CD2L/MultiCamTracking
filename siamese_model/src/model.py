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
class REID:
    def __init__(self, model:str = 'resnet50', weights:str = None, loss:str = 'softmax', dist_metric = 'euclidean'):
        self.use_gpu = torch.cuda.is_available()
        self.model = torchreid.models.build_model(
            name=model,
            num_classes=1,  # human
            loss=loss,
            pretrained=True,
            use_gpu=self.use_gpu
        )
        if weights is not None:
            torchreid.utils.load_pretrained_weights(self.model, weights)

        if self.use_gpu:
            self.model = self.model.cuda()
            _, self.transform_te = build_transforms(
                height=256, width=128,
                random_erase=False,
                color_jitter=False,
                color_aug=False
            )
        self.dist_metric = dist_metric
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
            features = features.data.cpu()
            f.append(features)
        f = torch.cat(f, 0)
        return f

    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        return distmat.numpy()
