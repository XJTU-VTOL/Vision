import sys
from pathlib import Path
import os
os.environ["PYTHONPATH"] = '/opt/data/private/Yolo/VisionPro'
from datasets import *
import cv2
import numpy

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 ,  # image rotation (+/- deg)
       'translate': 0.05 ,  # image translation (+/- fraction)
       'scale': 0.05 ,  # image scale (+/- gain)
       'shear': 0.641 }  # image shear (+/- deg)

def test_visdrone():
    """
    Test cases for visdrone datasets
    """
    train = {
        'name': 'VisDrone',
        'img_path': '/opt/data/private/VisDrone/VisDrone2019-DET-train/images',
        'label_path': '/opt/data/private/VisDrone/VisDrone2019-DET-train/annotations',
        'verbose': True
    }

    visdrone = VisDrone(train)
    img, labels = visdrone[15]

    img = img[:, :, ::-1]
    for label in labels:
        bbox = label['bbox']
        bbox = bbox.astype(np.int32)
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
    cv2.imwrite("visdrone.jpg", img)

    # lenth = len(visdrone)
    # cat = []
    # for idx in range(lenth):
    #     img, label = visdrone[idx]
    #     for label in labels:
    #         s = label['category_id']
    #         if s not in cat:
    #             cat.append(s)
    # print(cat)
    # print(lenth)

if __name__=='__main__':
    test_visdrone()