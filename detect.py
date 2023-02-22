import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import cv2 as cv
import random
from siamese_model.src.utils import distance
from siamese_model.src.model import SiameseModel, SMResNet101, torchereidModels
import torch
from torch import nn
import numpy as np
import shutil
from time import time 

parser = ArgumentParser()

parser.add_argument('-r', '--ref', type=str)
parser.add_argument('-i', '--image', type=str)
parser.add_argument('-n', '--n-samples', type=int, default=3)
parser.add_argument('-s', '--stride', type=int, default=3)

TEMP_PFOLDER = '.temp/'

torch.no_grad()

def extract_persons(folder: str, n_samples: int = 3, encoder = None):
    persons = os.listdir(folder)
    print('Nb persons', len(persons))

    imgs = []    
    imgs_enc = []    
    for pfolder_path in persons:
        pfolder = os.listdir(os.path.join(folder, pfolder_path))

        n_samples = min(n_samples, len(pfolder))

        pimgs = random.choices(pfolder, k=n_samples)

        img_batch = []
        img_batch_enc = []
        for pth in pimgs:
            im = cv.imread(os.path.join(folder, pfolder_path, pth), cv.IMREAD_ANYCOLOR)
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            im = cv.resize(im, (200,400))
            img_batch.append(im)
            if encoder is not None:
                im = encoder(im).squeeze().float().cpu()
            img_batch_enc.append(im)
                           
        imgs.append(img_batch)
        imgs_enc.append(img_batch_enc)
    return imgs_enc, imgs

def print_imgs(im_1, im_2, distance):
    fig, axes = plt.subplots(nrows=max(len(im_1), 2), ncols=3, figsize=(5,8)) 
    fig.tight_layout()
    for row, im in enumerate(im_1):
        ax = axes[row]
        ax[0].imshow(im)
        ax[0].axis('off')

        ax[1].imshow(im_2[row])
        ax[1].axis('off')

        ax[2].text(0,0,'%.2f'%distance[row])
        ax[2].axis('off')

    plt.savefig('similarity.jpeg')

def main(opt):
    # ref_img = cv.imread(opt['ref'], cv.IMREAD_ANYCOLOR)
    # ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2RGB)
    # ref_img = cv.resize(ref_img, (200,400))

    # Similarity model
    extractor = torchereidModels(weights='./siamese_model/checkpoints-saved/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')

    # YoloV8 detection
    print('Video #1')
    start = time()
    os.system(f'python ./yolov8_tracking/track.py --source {opt["image"]} --save-crop --project .temp --name v1 --vid-stride {opt["stride"]}')
    print('%.2f'%(time()-start), 's')

    print('Video #2')
    start = time()
    os.system(f'python ./yolov8_tracking/track.py --source {opt["ref"]} --save-crop --project .temp --name v2 --vid-stride {opt["stride"]}')
    print('%.2f'%(time()-start), 's')

    start = time()
    imgs_v1_enc, imgs_v1 = extract_persons(os.path.join(TEMP_PFOLDER, 'v1/crops/person'), opt['n_samples'], extractor)
    imgs_v2_enc, imgs_v2 = extract_persons(os.path.join(TEMP_PFOLDER, 'v2/crops/person'), opt['n_samples'], extractor)    
    print('%.2f'%(time()-start), 's')

    shutil.rmtree('.temp')
    
    # Calcul average similarity for each person
    distances = []
    for batch_idx, img1_batch in enumerate(imgs_v1_enc):
        img1_mean = torch.mean(torch.stack(img1_batch), 0)
        img1_mean = img1_mean[None, :]

        distance_res = []
        for batch2_idx, img2_batch in enumerate(imgs_v2_enc):            
            img2_mean = torch.mean(torch.stack(img2_batch), 0)
            img2_mean = img2_mean[None, :]

            distance_res.append(distance(img1_mean, img2_mean).item())
        
        distances.append(distance_res)

    dist = []
    imgs_v1_f = []
    imgs_v2_f = []

    for idx ,dist_val in enumerate(distances):
        print(distances)
        print(dist_val)
        if np.min(dist_val) < 300:
            imgs_v1_f.append(imgs_v1[idx][0])
            imgs_v2_f.append(imgs_v2[idx][0])
            dist.append(np.min(dist_val))

    # print('--------------------')
    # print('Folder ID (mean):',idx+1)
    # print('Val (mean):','%0.2f'%means[idx])
    # print('Folder ID (min):', min_idx+1)
    # print('Val (min):','%0.2f'%min_val)
    
    imgs_v2 = np.array([batch[0] for batch in imgs_v2])

    print('Nb persons detected:', len(dist))
    print_imgs(imgs_v1_f, imgs_v2_f, dist)

    # cv.imwrite('result-mean.jpg', cv.cvtColor(imgs[idx][0], cv.COLOR_RGB2BGR))
    # cv.imwrite('result-min.jpg', cv.cvtColor(imgs[min_idx][0], cv.COLOR_RGB2BGR))


if __name__ == '__main__':
    opt = parser.parse_args()    
    main(vars(opt))

