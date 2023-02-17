import os
from argparse import ArgumentParser
import cv2 as cv
import random
from siamese_model.src.model import SiameseModel, SMResNet101, MCT_VGG16
from siamese_model.src.utils import distance
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
DSIZE = (224,224)

torch.no_grad()

def main(opt):
    ref_img = cv.imread(opt['ref'], cv.IMREAD_ANYCOLOR)
    ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2RGB)
    ref_img = cv.resize(ref_img, DSIZE)

    # YoloV8 detection
    os.system(f'python ./yolov8_tracking/track.py --source {opt["image"]} --save-crop --project .temp --vid-stride {opt["stride"]} --retina-masks')

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
            im = cv.resize(im, DSIZE)
            img_batch.append(im)
                           
        imgs.append(img_batch)
    

    # Similarity model
    # model = SMResNet101()
    # rn101 = nn.DataParallel(model)
    # rn101 = rn101.to('cuda')

    # rn101.load_state_dict(torch.load('./siamese_model/checkpoints-saved/checkpoint_rn101_400.pkl')['model'])

    # load the model
    model = MCT_VGG16()
    rn101 = model

    ref_preprocessed = np.expand_dims(ref_img, 0)
    ref_preprocessed = model.preprocessing(ref_preprocessed)
    ref_tensor = torch.from_numpy(rn101(ref_preprocessed))

    # Calcul average similarity for each person
    means = []
    min_val = 1000
    min_idx = -1
    max_val = -1
    max_idx = -1


    for  batch_idx, img_batch in enumerate(imgs):
        img_batch = np.array(img_batch, dtype='float32')
        img_batch_preprocessed = model.preprocessing(img_batch)
        img_batch_tensors = rn101(img_batch_preprocessed)

        distance_res = []
        for img_tensor in img_batch_tensors:
            distance_res.append(distance(ref_tensor, torch.from_numpy(img_tensor), type='cosine').item())
            if distance_res[-1] < min_val:
                min_val = distance_res[-1]
                min_idx = batch_idx
            if distance_res[-1] > max_val:
                max_val = distance_res[-1]
                max_idx = batch_idx
        
        distance_res = np.array(distance_res)
        means.append(np.average(distance_res))


    idx = np.argmin(means)

    print('--------------------')
    print('Folder ID (mean):',idx+1)
    print('Val (mean):','%0.2f'%means[idx])
    print('Folder ID (min):', min_idx+1)
    print('Val (min):','%0.2f'%min_val)
    print('Folder ID (max):', max_idx+1)
    print('Val (max):','%0.2f'%max_val)

    shutil.rmtree('.temp')
    cv.imwrite('result-mean.jpg', cv.cvtColor(imgs[idx][0], cv.COLOR_RGB2BGR))
    cv.imwrite('result-min.jpg', cv.cvtColor(imgs[min_idx][0], cv.COLOR_RGB2BGR))
    cv.imwrite('result-max.jpg', cv.cvtColor(imgs[max_idx][0], cv.COLOR_RGB2BGR))



if __name__ == '__main__':
    opt = parser.parse_args()    
    main(vars(opt))

