# IMPORTS
import torch
import os 
import cv2 as cv

# CONST - Needs to be edit!
YOLO_PATH = './yolov5'
IMAGE_PATH = './utils/images-demo/rouen-centre-ville-1.jpg'

# CODE
img = cv.imread(IMAGE_PATH, cv.IMREAD_COLOR)

yolo = torch.hub.load(YOLO_PATH, "custom", path=os.path.join(YOLO_PATH, 'yolov5l.pt'), source="local")
yolo.conf = 0.6

bboxes = yolo(img)

if len(bboxes) > 0:
    # If yolo detect someone, we save the first result in the file system
    # The image sould be used later in the siamese model
    # 
    # cls 0 == person
    # cls 1 == bicycle
    # cls 2 == car
    #
    selected_bboxes = list(map(lambda x : x if x['cls'] == 0 else None, bboxes.crop(save=False, )))
    if len(selected_bboxes) > 1:
        cv.imwrite('bbox-saved.jpg', selected_bboxes[0]['im'])