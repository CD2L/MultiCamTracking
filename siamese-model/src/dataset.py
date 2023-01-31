from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os 
import cv2 as cv
import random
import torch

class ImageDataset(Dataset):
    def __init__(self, folder) -> None:
        self.folder = folder
        self.dataset = os.listdir(folder)
        
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.RandomPerspective(distortion_scale=0.04, p=0.4, fill=255),
            transforms.RandomPosterize(bits=3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.33), value=(0,0,0)),
        ])  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor = cv.imread(os.path.join(self.folder, self.dataset[index]), cv.IMREAD_ANYCOLOR)

        pos = self.transforms(anchor.copy())
        rdm_idx = index
        while rdm_idx == index:
            rdm_idx = random.randint(0, len(self.dataset)-1)
        neg = self.transforms(cv.imread(os.path.join(self.folder, self.dataset[rdm_idx]), cv.IMREAD_ANYCOLOR))
        
        neg = neg.float()
        pos = pos.float()

        anchor = torch.from_numpy(anchor/255)
        anchor = anchor.permute(2,0,1).float()

        return anchor, pos, neg