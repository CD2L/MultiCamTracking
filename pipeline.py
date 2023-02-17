import os
from argparse import ArgumentParser
import cv2 as cv
import random
from siamese_model.src.utils import distance
from siamese_model.src.model import SiameseModel, SMResNet101, torchereidModels
import torch
from torch import nn
import numpy as np
import shutil

parser = ArgumentParser()

parser.add_argument('-r', '--ref', type=str)
parser.add_argument('-i', '--image', type=str)
parser.add_argument('-n', '--n-samples', type=int, default=3)
parser.add_argument('-s', '--stride', type=int, default=3)

TEMP_PFOLDER = '.temp/exp/crops/person/'

torch.no_grad()

def main(opt):
    ref_img = cv.imread(opt['ref'], cv.IMREAD_ANYCOLOR)
    ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2RGB)
    ref_img = cv.resize(ref_img, (200,400))

    # YoloV8 detection
    os.system(f'python ./yolov8_tracking/track.py --source {opt["image"]} --save-crop --project .temp --vid-stride {opt["stride"]}')

    persons = os.listdir(TEMP_PFOLDER)
    print('Nb persons', len(persons))

    persons = sorted(persons)

    imgs = []    
    for pfolder_path in persons:
        pfolder = os.listdir(os.path.join(TEMP_PFOLDER, pfolder_path))

        n_samples = min(opt['n_samples'], len(pfolder))

        pimgs = random.choices(pfolder, k=n_samples)

        img_batch = []
        for pth in pimgs:
            im = cv.imread(os.path.join(TEMP_PFOLDER, pfolder_path, pth), cv.IMREAD_ANYCOLOR)
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            im = cv.resize(im, (200,400))
            img_batch.append(im)
                           
        imgs.append(img_batch)
    

    # Similarity model
    extractor = torchereidModels(weights='./siamese_model/checkpoints-saved/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')
    
    ref_tensor = extractor(ref_img)

    # Calcul average similarity for each person
    means = []
    min_val = 1000
    min_idx = -1

    for  batch_idx, img_batch in enumerate(imgs):
        img_batch = np.array(img_batch)

        distance_res = []
        for img_tensor in img_batch:
            img_ten = extractor(img_tensor)
            
            distance_res.append(distance(ref_tensor,img_ten).item())
            if distance_res[-1] < min_val:
                min_val = distance_res[-1]
                min_idx = batch_idx
        
        distance_res = np.array(distance_res)
        means.append(np.average(distance_res))


    idx = np.argmin(means)

    print('--------------------')
    print('Folder ID (mean):',idx+1)
    print('Val (mean):','%0.2f'%means[idx])
    print('Folder ID (min):', min_idx+1)
    print('Val (min):','%0.2f'%min_val)

    shutil.rmtree('.temp')
    cv.imwrite('result-mean.jpg', cv.cvtColor(imgs[idx][0], cv.COLOR_RGB2BGR))
    cv.imwrite('result-min.jpg', cv.cvtColor(imgs[min_idx][0], cv.COLOR_RGB2BGR))


if __name__ == '__main__':
    opt = parser.parse_args()    
    main(vars(opt))

