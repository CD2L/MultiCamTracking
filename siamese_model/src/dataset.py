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
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=255),
            transforms.RandomPosterize(bits=3,p=0.3),
            transforms.RandomEqualize(0.2),
            transforms.ColorJitter((0.5,1.2),(0.5,1.2),(0.5,1.2)),
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
    

class MultiAnglePersonDataset(Dataset):
    def __init__(self, folder, customTransforms=None) -> None:
        self.folder = folder
        self.dataset = os.listdir(folder)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPosterize(bits=3,p=0.3),
            transforms.ColorJitter((0.5,1.2),(0.5,1.2),(0.5,1.2)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.07), ratio=(0.3, 3.33), value=(0,0,0)),
        ])  

        if customTransforms is not None:
            self.transforms = customTransforms


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        #
        # file_name format: <character index>_c<camera index>s<image index>_<image index>_00.jpg 
        # example: 
        #       0001_c1s1_4982433_00.jpg
        #
        anchor_name = self.dataset[index]
        character_idx = anchor_name.split('_')[0]
        positive_list = [img for img in self.dataset if img.startswith(character_idx)]
        negative_list = [img for img in self.dataset if not img.startswith(character_idx)]

        anchor = cv.imread(os.path.join(self.folder, anchor_name), cv.IMREAD_ANYCOLOR)
        
        pos_name = random.choice(positive_list)
        neg_name = random.choice(negative_list)

        try:
            pos = self.transforms(cv.imread(os.path.join(self.folder, pos_name), cv.IMREAD_ANYCOLOR))
            neg = self.transforms(cv.imread(os.path.join(self.folder, neg_name), cv.IMREAD_ANYCOLOR))
        except:
            print('neg:',neg_name)
            print('pos:',pos_name)
        
        neg = neg.float()
        pos = pos.float()

        anchor = torch.from_numpy(anchor/255)
        anchor = anchor.permute(2,0,1).float()

        return anchor, pos, neg
