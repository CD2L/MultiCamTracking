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
from time import time, sleep
from math import floor, ceil

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

def upsampling_bbox(nb_frames, tracking, stride):

    tracking_by_id = {}
    # Isolate every person
    for index, track in enumerate(tracking):
        id = int(track[1])
        if id not in tracking_by_id.keys():
            tracking_by_id[id] = []
        tracking_by_id[id].append(list(map(lambda x : floor(float(x)), track[:6])))

    tracking_by_frame = {}
    for i in range(nb_frames+1):
        tracking_by_frame[i] = []

    for key, person in tracking_by_id.items():
        for i in range(len(person)):
            x0, y0, h0, w0 = person[i][2:6]

            if i+1 < len(person):
                x1, y1, h1, w1 = person[i+1][2:6]
                next_frame = person[i+1][0]
            else : 
                x1 = None

            current_frame = person[i][0]
            video_frame = current_frame * stride

            if video_frame > nb_frames:
                raise ValueError(f'video_frame ({video_frame}) is bigger than nb_frames ({nb_frames})')

            if current_frame+1 == next_frame and x1 is not None:
                for j in range(stride):
                    tracking_by_frame[video_frame+j].append([
                            key,
                            floor(x0+(j/stride)*(x1-x0)), 
                            floor(y0+(j/stride)*(y1-y0)), 
                            floor(h0+(j/stride)*(h1-h0)), 
                            floor(w0+(j/stride)*(w1-w0))
                    ])
            else:
                tracking_by_frame[video_frame].append([
                    key,
                    floor(x0), 
                    floor(y0), 
                    floor(h0), 
                    floor(w0)
                ])
    return tracking_by_frame

def count_frames(video_path):
    cap = cv.VideoCapture(video_path)
    c = 0

    while(True):
        ret, frame = cap.read()
        if(ret):
            c += 1
        else:
            print(c)
            return c


def video_tracking(video_path, tracking: dict, stride = 1, corres:dict = None):
    cap = cv.VideoCapture(video_path)
    output = cv.VideoWriter('v2_output.mp4', cv.VideoWriter_fourcc(*'MP4V'), 30, (720, 1080))

    for _, tracks in tracking.items():
            ret, frame = cap.read()
            if(ret):
                for bbox in tracks:
                    if (corres is None) or (corres is not None and int(bbox[0]) in corres.keys()):
                        cv.rectangle(frame,
                            (bbox[1], bbox[2]),
                            (bbox[1]+bbox[3], bbox[2]+bbox[4]), 
                            (0, 255, 0), 2)
                        id = bbox[0]
                        if corres is not None:
                            id = corres[int(bbox[0])]
                        cv.putText(frame, str(id), (bbox[1], bbox[2]-15), cv.FONT_HERSHEY_SIMPLEX, 1, color=(0,255,0), thickness=2)

                sleep(0.02)
                output.write(frame)

                cv.imshow("output", frame)
                if cv.waitKey(1) & 0xFF == ord('s'):
                    break
            else:
                break

    cv.destroyAllWindows()
    output.release()
    cap.release()


def main(opt):
    # Similarity model
    reid = REID(opt['model'], opt['weights'], dist_metric='cosine')
    
    if not os.path.exists('.temp') or not opt['cache']:
        # YoloV8 detection and tracking
        print('Video #1')
        start = time()
        os.system(f'python ./yolov8_tracking/track.py --source {opt["image"]} --save-crop --save-txt --project .temp --name v1 --vid-stride {opt["stride"]} --tracking-method strongsort')
        print('%.2f'%(time()-start), 's')

        print('Video #2')
        start = time()
        os.system(f'python ./yolov8_tracking/track.py --save-vid --source {opt["ref"]} --save-crop --project .temp --name v2 --vid-stride {opt["stride"]} --tracking-method strongsort')
        print('%.2f'%(time()-start), 's')

    start = time()
    imgs_v1_enc, imgs_v1 = extract_persons(os.path.join(TEMP_PFOLDER, 'v1/crops/person'), opt['n_samples'], reid)
    imgs_v2_enc, imgs_v2 = extract_persons(os.path.join(TEMP_PFOLDER, 'v2/crops/person'), opt['n_samples'], reid)    
    print('%.2f'%(time()-start), 's')


    txt_file = os.path.join(TEMP_PFOLDER, 'v1/tracks/', opt['image'].split('\\')[-1].split('.')[0]+'.txt')
    tracks = []
    # Keep the tracks file 
    with open(txt_file, 'r') as f:
        for line in f:
            tracks.append(line.split(' '))


    ########## TEMPORARY
    # tmp = []
    # for track in tracks:
    #     step = track[1:6]
    #     tmp.append(step)

    # tracking = tmp
    # del tmp
    # ###################

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
    corres = {}

    for idx ,dist_val in enumerate(distances):
        # Preventing from returning an already reidentified person
        dist_val = [val if id not in corres else 99999 for id, val in enumerate(dist_val)]
        min_idx = np.argmin(dist_val)
        
        if np.min(dist_val) < opt['threshold']:
            corres[np.argmin(dist_val)] = idx
            imgs_v1_f.append(imgs_v1[idx][0])
            imgs_v2_f.append(imgs_v2[min_idx][0])
            dist.append(np.min(dist_val))

    imgs_v2 = np.array([batch[0] for batch in imgs_v2])

    if opt['verbose']:
        # Printing time !
        print('Nb persons detected:', len(dist))
        print('--Correspondance--')
        print('v1 , v2')
        for idx_1, idx_2 in corres.items():
            print(idx_2,',',idx_1)
        print_imgs(imgs_v1_f, imgs_v2_f, dist)
        tracking = upsampling_bbox(count_frames(opt['image']), tracks, opt['stride'])
        video_tracking(opt['image'], tracking, opt['stride'], corres)


    return corres

if __name__ == '__main__':
    
    opt = parser.parse_args()    
    main(vars(opt))

