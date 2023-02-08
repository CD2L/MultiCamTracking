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

ap.add_argument("-i", "--image", required=True,
   help="Image compared to the reference image in the simalirity comparison phase")
ap.add_argument("-r", "--ref",
   help="Reference image")
ap.add_argument("-o", "--output",
   help="Path of the output image")


args = vars(ap.parse_args())

def is_the_same_box(predicted: np.ndarray, original: np.ndarray) -> bool:
    return False

def extract_bboxes(bboxes, cls = 0, bbox_dim = None, save_dir = None):
    selected_bboxes = list(map(lambda x : x if x['cls'] == cls else None, bboxes.crop(save=False, )))
    selected_bboxes = [i for i in selected_bboxes if i is not None]

    bboxes = {"im": [], "box": []}

    for i, bbox in enumerate(selected_bboxes):
        if bbox is not None:
            bbox_img = bbox['im']
            if bbox_dim is not None:
                bbox_img = cv.resize(bbox['im'], bbox_dim)
            
            bboxes["im"].append(bbox_img)
            bboxes["box"].append(bbox["box"])

            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                cv.imwrite(os.path.join(save_dir,f"bbox{i}.jpg"), bbox_img)
    
    return bboxes

def draw_bbox(box, img, output=None):
    box_ = [int(i) for i in box]
    img_and_box = cv.rectangle(
        img,
        pt1=(box_[0], box_[1]),
        pt2=(box_[2], box_[3]),
        color=(255,0,0),
        thickness=2
    )

    img_and_box = cv.cvtColor(img_and_box, cv.COLOR_BGR2RGB)
    if output is not None:
        cv.imwrite(output, img_and_box)

def main():
    
    accuracy = 0
    distance_mean = 0

    img = cv.imread(args['image'], cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    yolo = torch.hub.load(YOLO_PATH, "custom", path=os.path.join(YOLO_PATH, 'yolov5l.pt'), source="local")
    yolo.conf = 0.35
    yolo.iou = 0.8

    weights = torch.load('./siamese_model/checkpoints-saved/checkpoint_exp37_200.pkl')

    model = SiameseModel()
    siamese = nn.DataParallel(model)
    siamese = siamese.to('cuda')

    siamese.load_state_dict(weights['model'])

    bboxes = yolo(img)
    print("Nb of bboxes: ", len(bboxes.crop(save=False, )))

    if len(bboxes) > 0:
        # If yolo detect someone, we save the first result in the file system
        # The image sould be used later in the siamese model
        # 
        # COCO configuration :
        #    cls 0 == person
        #    cls 1 == bicycle
        #    cls 2 == car
        #

        refs = extract_bboxes(bboxes, cls=0, bbox_dim=(64,128))

        print("Nb of person: ", len(refs['im']))
        if len(refs['im']) > 1:
            # Encoding the ref images

            refs_image = model.preprocessing(np.array(refs['im']))
            refs_image = siamese(refs_image)

            compared_img = []
            for im in refs['im']:
                im = cv.flip(im,1)
                compared_img.append(im)
            compared_img = model.preprocessing(np.array(compared_img))
            compared_img = siamese(compared_img)

            for idx, ref in enumerate(refs_image):
                ref = ref[None, :]
                distance_res = np.array([distance(ref, o_bbox).item() for o_bbox in compared_img])

                distance_mean += distance_res[np.argmin(distance_res)]

                if np.argmin(distance_res) == idx:
                    accuracy += 1
                            
            accuracy /= len(refs['im'])
            distance_mean /= len(refs['im'])

            print('Accuracy', '%.2f'%accuracy)
            print('Mean distance', '%.2f'%distance_mean)

if __name__ == '__main__':
    main()
