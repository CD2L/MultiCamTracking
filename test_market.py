# IMPORTS
import torch
import os 
import cv2 as cv
from siamese_model.src.model import SiameseModel
from siamese_model.src.utils import distance
from torch import nn
from tqdm import tqdm
import numpy as np

def main():
    
    accuracy = 0
    distance_mean_pos = 0
    nb_pos = 0
    distance_mean_neg = 0
    nb_neg = 0

    # Preparation
    query = os.listdir('./Market-1501-v15.09.15/query')
    anchors = []

    for i, filename in enumerate(query):
        if i == 500: break
        flag = False
        for anchor in anchors:
            if anchor.startswith(filename.split('_')[0]):
                flag = True
                break
        if not flag:
            anchors.append(filename)
    
    anchors = sorted(anchors)
    print("Anchor size", len(anchors))

    gt_bbox = os.listdir('./Market-1501-v15.09.15/gt_bbox')
    validation_data = []
    character_list = list(map(lambda x: x.split('_')[0], anchors))
    for filename in tqdm(gt_bbox, 'dataset creation 1/2'):
        current_character_idx = filename.split('_')[0]
        flag = False

        if current_character_idx not in character_list:
            flag = True

        if not flag:
            for file in validation_data:
                if file.startswith(current_character_idx):
                    flag = True
                    break
            if not flag:
                validation_data.append(filename)

    validation_data = sorted(validation_data)

    anchor_img = []
    for img in anchors:
        if not img.endswith('db'):
            img = cv.imread(os.path.join('./Market-1501-v15.09.15/query', img), cv.IMREAD_ANYCOLOR)
            img = cv.resize(img, (68,128))
            anchor_img.append(img)

    validation_img = []
    for img in tqdm(validation_data, 'dataset creation 2/2'):
        if not img.endswith('db'):
            img = cv.imread(os.path.join('./Market-1501-v15.09.15/gt_bbox', img), cv.IMREAD_ANYCOLOR)
            img = cv.resize(img, (68,128))
            validation_img.append(img)

    # Tests
    model = SiameseModel()
    sm = nn.DataParallel(model)
    sm = sm.to('cuda')
    
    sm.load_state_dict(torch.load('./siamese_model/checkpoints-saved/checkpoint_exp37_200.pkl')['model'])

    anchor_img = model.preprocessing(np.array(anchor_img, np.float32))
    validation_img = model.preprocessing(np.array(validation_img, np.float32))

    tensor_anchors = sm(anchor_img)
    tensor_val = sm(validation_img)
    
    for anchor_idx, anchor in enumerate(tqdm(anchors)):
        character_idx = anchor.split('_')[0]
        tensor_anchor = tensor_anchors[anchor_idx]

        distance_res = []
        for val_idx, val_file in enumerate(validation_data):
            distance_res.append(distance(tensor_anchor, tensor_val[val_idx][None, :]).item())
        
        distance_res = np.array(distance_res)
        idx = np.argmin(distance_res)

        if validation_data[idx].startswith(character_idx):
            accuracy += 1
            distance_mean_pos += distance_res[idx]
            nb_pos += 1
        
        for idx_res, res in enumerate(distance_res):
            if idx_res != idx:
                distance_mean_neg += res
                nb_neg += 1

    accuracy /= len(anchors)
    distance_mean_pos /= nb_pos
    distance_mean_neg /= nb_neg

    print('\n')
    print('Accuracy', accuracy)
    print('Distance (+)', distance_mean_pos)
    print('Distance (-)', distance_mean_neg)


if __name__ == '__main__':
    main()
