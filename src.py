# IMPORTS
import argparse
import torch
import os 
import cv2 as cv
from siamese_model.src.model import SiameseModel
from siamese_model.src.utils import distance
from torch import nn
import numpy as np


# CONST - Needs to be edit!
YOLO_PATH = './yolov5'
IMAGE_PATH = './utils/images-demo/rouen-centre-ville-1.jpg'

ap = argparse.ArgumentParser()

ap.add_argument("-r", "--ref", required=True,
   help="Reference image, anchor")
ap.add_argument("-i", "--image", required=True,
   help="Image compared to the reference image in the simalirity comparison phase")

args = vars(ap.parse_args())

def draw_bbox(box, img):
    box_ = [int(i) for i in box]
    img_and_box = cv.rectangle(
        img,
        pt1=(box_[0], box_[1]),
        pt2=(box_[2], box_[3]),
        color=(255,0,0),
        thickness=2
    )

    img_and_box = cv.cvtColor(img_and_box, cv.COLOR_BGR2RGB)
    cv.imwrite('result.jpg', img_and_box)

def main():
    # CODE
    img = cv.imread(args['image'], cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    ref = cv.imread(args['ref'], cv.IMREAD_ANYCOLOR)
    ref = cv.cvtColor(ref, cv.COLOR_BGR2RGB)
    ref = cv.resize(ref, (64,128))
    ref = ref.reshape((1,*ref.shape))


    yolo = torch.hub.load(YOLO_PATH, "custom", path=os.path.join(YOLO_PATH, 'yolov5l.pt'), source="local")
    yolo.conf = 0.35

    weights = torch.load('./siamese_model/checkpoints-saved/checkpoint_exp5_200.pkl')

    model = SiameseModel()
    siamese = nn.DataParallel(model)
    siamese = siamese.to('cuda')

    siamese.load_state_dict(weights['model'])

    bboxes = yolo(img)
    print("Nb of bboxes: ", len(bboxes))
    if len(bboxes) > 0:
        # If yolo detect someone, we save the first result in the file system
        # The image sould be used later in the siamese model
        # 
        # COCO configuration :
        #    cls 0 == person
        #    cls 1 == bicycle
        #    cls 2 == car
        #
        selected_bboxes = list(map(lambda x : x if x['cls'] == 0 else None, bboxes.crop(save=False, )))
        selected_bboxes = [i for i in selected_bboxes if i is not None]
        print("Nb of person: ", len(selected_bboxes))
        if len(selected_bboxes) > 1:
            ref = model.preprocessing(ref)
            o_ref = siamese(ref)

            bboxes = []
            for i, bbox in enumerate(selected_bboxes):
                if bbox is not None:
                    bbox_img = cv.resize(bbox['im'],(64,128))
                    #cv.imwrite(f'ref_d{i}-resized.jpg', bbox_img)

                    bboxes.append(bbox_img)
            bboxes = np.array(bboxes)
            o_selected_bboxes = siamese(model.preprocessing(bboxes))

            distance_res = np.array([distance(o_ref, o_bbox).item() for o_bbox in o_selected_bboxes])
            bbox = selected_bboxes[np.argmin(distance_res)]['box']

            print('Index of img: ', np.argmin(distance_res))
            print('Distance: ', distance_res[np.argmin(distance_res)])
            draw_bbox([i.item() for i in bbox], img)

if __name__ == '__main__':
    main()
