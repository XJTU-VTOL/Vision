import glob
import math
import string
import os
import random
import shutil
import time
import argparse
import matplotlib
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from utils.common_utils import xyxy2xywh, xywh2xyxy

from utils.dataset_utils import calculate_corners
from utils.datasets import letterbox, random_affine, KITTI, COCO, VisDrone
  

BaseDataset = {
    "COCO": COCO,
    "KITTI": KITTI,
    "VisDrone": VisDrone
}

def create_dataset(config: dict):
    """
    To add other datasets, please follow the rules below.
    The dataset should take a disctionary as input, which includes `img_path`, `label_path` and other keys as parameters. 
    Please specify these parameters in the documentation of your own dataset so that we can parse what you want.
    The dictionary must contain the `name` key , which indicates the base dataset. Add the name and dataset in 
    the above `BaseDataset` dictionary.

    your dataset should return:
    image in np.array format:
        Note: dtype must be np.uint8 and the channel should be in `RGB` (H, W, 3)
    a dictionary with keys 
        "bbox": (top_left_x, top_left_y, width, height) 
        "image_id": corresponding image id/name
        "category_id": category_id of the bounding box
                    range (1 ~ nc)
    """
    return BaseDataset[config["name"]](config)

class LoadDataset(Dataset):
    """
    Input configuration dictionary must contain

    `base` : (dictionary) parameters for the base dataset. Please follow the documentation of the function
             `create_dataset` to set up the dictionary.
    `image_size`: (tuple or int) the input image size
    `hyper`: (dictionary) parameters for image augmentation, which must contain:
             `degrees`:  the amplitutde of random rotation degrees for image.
             `translate`:  the amplitutde of random translation for image.
             `scale`: the amplitutde of random scale for image.
             `shear`: random shear scale
    `augment`: whether to augment the image

    The augmentation includes random affine transformation, left-right flip and up-down flip with 0.5 probability.

    Return:
        img: torch.Tensor float, normalied in 0.~1. in RGB order
        label: (6, ) (batch_id, cls_id, x, y, w, h)
        img_id: the name of the img file
        shape: (h0, w0), ((h / h0, w / w0), pad) (original shape, (resize ratio), padding)

    TODO:
        ADD hsv and other augmentation
    """
    def __init__(self, config: dict):
        super(LoadDataset, self).__init__()
        self.dataset = create_dataset(config['base'])
        self.img_size = config["image_size"]
        self.augment = config["augment"]
        if self.augment:
            assert 'hyper' in config.keys(), 'Please add the parameters for data augmentation !'
            self.hyp = config['hyper']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, labels = self.dataset[index]
        if type(img) is not np.ndarray:
            img = np.array(img)
        h0, w0 = img.shape[:2] # original size
        coors = np.array([label['bbox'] for label in labels]) # x_left, y_left, width, height  (left_up_corner + width, height)
        # assert len(coors.shape)==2, 'loaded empty label %d'%(index)
        img_id = np.array([label['image_id'] for label in labels])
        cls_id = np.array([label['category_id'] for label in labels], dtype=np.float32) - 1 # convert to 0 ~ 89

        img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.augment)
        h, w = img.shape[:2] # current shape
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # convert width, height 
        labels = coors.copy()
        nL = len(labels)  # number of labels
        if len(labels.shape) == 2:
            # convert to center coordinate
            labels[:, 0] = ratio[0] * (coors[:, 0] + coors[:, 2] / 2 ) + pad[0]  # pad width
            labels[:, 1] = ratio[1] * (coors[:, 1] + coors[:, 3] / 2 ) + pad[1]  # pad height
            

            # scale in current image
            labels[:, 2] = ratio[0] * coors[:, 2]
            labels[:, 3] = ratio[0] * coors[:, 3] 

            if self.augment:
                # convert from (center + w, h) to (x1y1x2y2)
                diag = xywh2xyxy(labels)
                diag = np.concatenate([cls_id[:, np.newaxis], diag], axis=1) # (N, cls + xyxy)

                img, diag = random_affine(img, diag,
                                        degrees=self.hyp['degrees'],
                                        translate=self.hyp['translate'],
                                        scale=self.hyp['scale'],
                                        shear=self.hyp['shear'])
                
                # re-assign the labels
                labels = xyxy2xywh(diag[:, 1:]) #(N, xywh)
                cls_id = diag[:, 0]

                # random left-right flip
                lr_flip = True
                if lr_flip and random.random() < 0.5:
                    img = np.fliplr(img)
                    if nL:
                        labels[:, 0] = w - labels[:, 0]

                # random up-down flip
                ud_flip = False
                if ud_flip and random.random() < 0.5:
                    img = np.flipud(img)
                    if nL:
                        labels[:, 1] = h - labels[:, 1]

            # Normalize coordinates 0 - 1 (For Yolo training)
            labels[:, [1, 3]] /= img.shape[0]  # height
            labels[:, [0, 2]] /= img.shape[1]  # width
            
            labels[np.where(labels>1.0)] = 1.0
            labels[np.where(labels<0.0)] = 0.0 

        nL = len(labels)
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 2:] = torch.from_numpy(labels)
            labels_out[:, 1] = torch.from_numpy(cls_id)

        # Convert
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img).float()/255.0, labels_out, img_id, shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
 
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