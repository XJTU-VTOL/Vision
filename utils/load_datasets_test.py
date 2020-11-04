import sys
from pathlib import Path
import os
print(Path(__file__).resolve().parent.parent)
sys.path.append(Path(__file__).resolve().parent.parent)
from load_datasets import *

if __name__=='__main__':
    # Feel Free to add your test code here !

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
       'degrees': 10,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

    config = {
        "base":{
            "name": "KITTI",
            "img_path": '/data/cxg1/VoxelNet_pro/Data/training/image_2',
            "label_path": '/data/cxg1/VoxelNet_pro/Data/training/label_2',
            "ids": "/data/cxg1/VoxelNet_pro/training/train.txt"
        },
        "img_size": 416,
        "hyper": hyp,
        "augment": True
    }

    kitti = LoadDataset(config)
    img, labels, _, _ = kitti[2]
    img = img.permute(1, 2, 0).numpy()
    labels = labels.numpy()
    h, w = img.shape[:2]
    for label in labels:
        top = (int((label[2]+label[4]/2)*w), int((label[3]+label[5]/2)*h))
        down = (int((label[2]-label[4]/2)*w), int((label[3]-label[5]/2)*h))
        img = cv2.rectangle(img, top, down, (255, 0, 0))
        font_scale = 0.03*int((label[4]*w))
        cv2.putText(img, "car", (int((label[2]-label[4]/2)*w), int((label[3]-label[5]/2)*h)), cv2.FONT_HERSHEY_PLAIN,  font_scale, (0,255,0), 1)

    img = cv2.imwrite("rotation.png", img)     