import os
from argparse import ArgumentParser
import cv2 as cv
import random
from siamese_model.src.model import SiameseModel, SMResNet101
from siamese_model.src.utils import distance
import torch
from torch import nn
import numpy as np
import shutil

parser = ArgumentParser()

parser.add_argument('-r', '--ref', type=str)
parser.add_argument('-i', '--image', type=str)
parser.add_argument('-n', '--n-samples', type=int, default=3)

TEMP_PFOLDER = '.temp/exp/crops/person/'

torch.no_grad()

def main(opt):
    ref_img = cv.imread(opt['ref'], cv.IMREAD_ANYCOLOR)
    ref_img = cv.resize(ref_img, (100,200))
    #global_img = cv.imread(opt['image'], cv.IMREAD_ANYCOLOR)

    # YoloV8 detection
    os.system(f'python ./yolov8_tracking/track.py --source {opt["image"]} --save-crop --project .temp --vid-stride 3')

    persons = os.listdir(TEMP_PFOLDER)
    print('Nb persons', len(persons))

    persons = sorted(persons)

    imgs = []    
    for pfolder_path in persons:
        pfolder = os.listdir(os.path.join(TEMP_PFOLDER ,pfolder_path))

        n_samples = min(opt['n_samples'], len(pfolder))
        pimgs = random.choices(pfolder, k=n_samples)
        pimgs = [cv.resize(cv.imread(os.path.join(TEMP_PFOLDER, pfolder_path, pth), cv.IMREAD_ANYCOLOR), (100,200)) for pth in pimgs]
        imgs.append(pimgs)
    
    # Calcul average similarity for each person
    model = SMResNet101()
    rn101 = nn.DataParallel(model)
    rn101 = rn101.to('cuda')

    rn101.load_state_dict(torch.load('./siamese_model/checkpoints-saved/checkpoint_rn101_400.pkl')['model'])
    
    print(ref_img.shape)
    ref_preprocessed = np.expand_dims(ref_img, 0)
    ref_preprocessed = model.preprocessing(ref_preprocessed)
    ref_tensor = rn101(ref_preprocessed)

    means = []
    for img_batch in imgs:
        img_batch = np.array(img_batch, dtype='float32')
        img_batch_preprocessed = model.preprocessing(img_batch)
        img_batch_tensors = rn101(img_batch_preprocessed)

        distance_res = []
        for img_tensor in img_batch_tensors:
            distance_res.append(distance(ref_tensor,img_tensor).item())
        
        distance_res = np.array(distance_res)
        means.append(np.average(distance_res))

    print(means)
    print('Folder ID:',np.argmin(means)+1)
    shutil.rmtree('.temp')

        

            
            
            

if __name__ == '__main__':
    opt = parser.parse_args()    
    main(vars(opt))

