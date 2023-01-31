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
            transforms.RandomPerspective(distortion_scale=0.2, p=0.7, fill=255),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.33), value=(0,0,0)),
            transforms.ConvertImageDtype(torch.float),
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
        
        anchor = torch.from_numpy(anchor)
        anchor = anchor.permute(2,0,1)

        return anchor, pos, neg