import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import cv2 as cv
import random
from siamese_model.src.utils import distance
from siamese_model.src.model import SiameseModel, SMResNet101, REID
import torch
import numpy as np
import shutil
from time import time 

parser = ArgumentParser()

parser.add_argument('-r', '--ref', type=str)
parser.add_argument('-i', '--image', type=str)
parser.add_argument('-n', '--n-samples', type=int, default=3)
parser.add_argument('-s', '--stride', type=int, default=3)
parser.add_argument('-c', '--cache', type=bool, default=False)
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--model', type=str, default='osnet_ain_x1_0')
parser.add_argument('-t', '--threshold', type=float, default=600)
parser.add_argument('-v', '--verbose', type=bool, default=True)

TEMP_PFOLDER = '.temp/'

torch.no_grad()

def extract_persons(folder: str, n_samples: int = 3, encoder:REID = None):
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
            img_batch_enc = encoder._features(img_batch)

        imgs.append(img_batch)
        imgs_enc.append(img_batch_enc)
    return imgs_enc, imgs

def print_imgs(im_1, im_2, distance):
    fig, axes = plt.subplots(nrows=max(len(im_1), 2), ncols=3, figsize=(5,20)) 
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
    # Similarity model
    reid = REID(opt['model'], opt['weights'], dist_metric='cosine')
    
    if not os.path.exists('.temp') or not opt['cache']:
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
    imgs_v1_enc, imgs_v1 = extract_persons(os.path.join(TEMP_PFOLDER, 'v1/crops/person'), opt['n_samples'], reid)
    imgs_v2_enc, imgs_v2 = extract_persons(os.path.join(TEMP_PFOLDER, 'v2/crops/person'), opt['n_samples'], reid)    
    print('%.2f'%(time()-start), 's')

    if not opt['cache']:
        shutil.rmtree('.temp')
    
    # Calcul average similarity for each person
    distances = []
    for img1_batch in imgs_v1_enc:
        distance_res = []
        for img2_batch in imgs_v2_enc:  
            distance_res.append(np.mean(reid.compute_distance(img1_batch, img2_batch)))
        
        distances.append(distance_res)

    dist = []
    imgs_v1_f = []
    imgs_v2_f = []
    corres = []

    for idx ,dist_val in enumerate(distances):
        # Preventing from returning an already reidentified person
        dist_val = [val if id not in [i for _,i in corres] else 99999 for id, val in enumerate(dist_val)]
        min_idx = np.argmin(dist_val)
        
        if np.min(dist_val) < opt['threshold']:
            corres.append((idx, np.argmin(dist_val)))
            imgs_v1_f.append(imgs_v1[idx][0])
            imgs_v2_f.append(imgs_v2[min_idx][0])
            dist.append(np.min(dist_val))

    imgs_v2 = np.array([batch[0] for batch in imgs_v2])

    if opt['verbose']:
        # Printing time !
        print('Nb persons detected:', len(dist))
        print('--Correspondance--')
        print('v1 , v2')
        for idx_1, idx_2 in corres:
            print(idx_1,',',idx_2)
        print_imgs(imgs_v1_f, imgs_v2_f, dist)

    return corres

if __name__ == '__main__':
    opt = parser.parse_args()    
    main(vars(opt))

