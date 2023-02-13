import os
from argparse import ArgumentParser
import cv2 as cv
import random
from siamese_model.src.model import SiameseModel, SMResNet101
import torch
from torch import nn

parser = ArgumentParser()

parser.add_argument('-r', '--ref', type=str)
parser.add_argument('-i', '--image', type=str)
parser.add_argument('-n', '--n-samples', type=int, default=3)

TEMP_PFOLDER = '.temp/exp/crops/person/'

def main(opt):
    #ref_img = cv.imread(opt['ref'], cv.IMREAD_ANYCOLOR)
    #global_img = cv.imread(opt['image'], cv.IMREAD_ANYCOLOR)

    # YoloV8 detection
    #os.system(f'python ./yolov8_tracking/track.py --source {opt["image"]} --save-crop --project .temp')

    persons = os.listdir(TEMP_PFOLDER)
    print('Nb persons', len(persons))

    imgs = []    
    for pfolder_path in persons:
        pfolder = os.listdir(os.path.join(TEMP_PFOLDER ,pfolder_path))

        n_samples = min(opt['n_samples'], len(pfolder))
        pimgs = random.choices(pfolder, k=n_samples)
        pimgs = [os.path.join(TEMP_PFOLDER, pfolder_path, pth) for pth in pimgs]
        imgs.append(pimgs)
    
    print(imgs)

    # Calcul average similarity for each person
    model = SMResNet101()
    rn101 = nn.DataParallel(model)
    rn101 = rn101.to('cuda')

    rn101.load_state_dict(torch.load('./siamese_model/checkpoints-saved/checkpoint_en101_400.pkl')['model'])
            
            

if __name__ == '__main__':
    opt = parser.parse_args()    
    main(vars(opt))

