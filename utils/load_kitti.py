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
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.datasets import CocoDetection
import random

from dataset_utils import calculate_corners


class2id = {
    'Car': 1,
    'Van': 2,
    'Truck': 3,
    'Pedestrian': 4,
    'Person_sitting': 5,
    'Cyclist': 6,
    'Tram': 7,
    'Misc': 0
}

def letterbox(img, new_shape=(416, 416), color=(0,0,0), auto=True, scaleFill=False, scaleup=True):
    """
        img: Input img (after resize, need to pad)
        new_shape: input shape for the neural network.
        color: padding color
    """
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(float(new_shape[0]) / shape[0], float(new_shape[1]) / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def calculate_corners(bbox) :
    """
        bbox: numpy array (N, 4) bbox representations in center_x, center_y, width, height

        return :
            numpy array (N, 4, 2):
                left_top_x, left_top_y, 
                right_top_x, right_top_y, 
                left_bottom_x, left_bottom_y, 
                right_bottom_x, right_bottom_y 
    """
    left_top = bbox[:, [0, 1]] - bbox[:, [2, 3]] / 2
    right_top = np.stack([bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] - bbox[:, 3] / 2], axis=1)
    left_bottom = np.stack([bbox[:, 0] - bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2], axis=1)
    right_bottom = bbox[:, [0, 1]] + bbox[:, [2, 3]] / 2

    return np.stack([left_top, right_top, left_bottom, right_bottom], axis=1)

def get_args():
    parser = argparse.ArgumentParser('LoadKitti')
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--img_path', type=str, default='/data/cxg1/VoxelNet_pro/Data/training/image_2')
    parser.add_argument('--label_path', type=str, default='/data/cxg1/VoxelNet_pro/Data/training/label_2')
    parser.add_argument('--augment', type=bool, default=True)

    args = parser.parse_args()
    return args


class LoadKitti(Dataset):
    def __init__(self, img_path, label_path, ids, img_size, hyp, augment, transform=None, target_transform=None):
        super(LoadKitti, self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.img_size = img_size
        self.augment = augment
        self.transform = transform

        with open(ids, 'r') as f:
            ids_file = f.readlines()
        id_num = [int(id_) for id_ in ids_file]

        img_list = sorted(os.listdir(img_path))
        label_list = sorted(os.listdir(label_path))

        self.img_list = []
        self.label_list = []

        for num in id_num:
            self.img_list.append(img_list[num])
            self.label_list.append(label_list[num])
        self.hyp = hyp

    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, item):
        """
        input: 
            item : img_list[item]

        return:
            img : torch.Tensor ()
            labels : torch.Tensor( format: order, class, x_center, y_center, width, hight)
        """
        #get image and labels
        img_place = self.img_list[item]
        label_place = self.label_list[item]
        img = cv2.imread(os.path.join(self.img_path, img_place))
        img = np.array(img, dtype = float)
        label_file = open(os.path.join(self.label_path, label_place), 'r')
        label = label_file.readlines()
        labels = []
        cls_id = []

        for fields in label: 
            fields=fields.strip()
            fields=fields.split(" ")
            cls_name = fields[0]
            cls_id.append(class2id[cls_name])
            labels.append(fields[1:])
        labels = np.array(labels)
        cls_id = np.array(cls_id)
        #scale to 416 by 416
        h0, w0 = img.shape[:2]
        # it is ok to just use letterbox 
        # r = self.img_size / max(h0, w0)  # resize image to img_size
        # if r != 1:  # always resize down, only resize up if training with augmentation
        #     interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w =  img.shape[:2] # resize shape

        img, padding_ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        #scale bbox
        bbox = np.array(labels[:, [3, 4, 5, 6]])
        bbox = bbox.astype(np.float64)
        nL = len(bbox)
        bbox[:, [0,2]] = bbox[:, [0,2]]*padding_ratio[0] + pad[0]
        bbox[:, [1,3]] = bbox[:, [1,3]]*padding_ratio[1] + pad[1]
        bbox_center = bbox.copy()

        # bbox center (x_center, y_center, w, h )
        bbox_center[:, 0] = ( bbox[:, 0] + bbox[:, 2] ) / 2
        bbox_center[:, 1] = ( bbox[:, 1] + bbox[:, 3] ) / 2
        bbox_center[:, 2] = bbox[:, 2] - bbox[:, 0]
        bbox_center[:, 3] = bbox[:, 3] - bbox[:, 1]  
        print(bbox_center)

        # rotation
        degrees = random.randint(-self.hyp['degrees'], self.hyp['degrees'])
        center = (img.shape[1]*0.5, img.shape[0]*0.5)
        Mat = cv2.getRotationMatrix2D(center, degrees, 1)
        img = cv2.warpAffine(img, Mat, (img.shape[1], img.shape[0]))
        corners = calculate_corners(bbox_center) # N, 4, 2
        for idx, item in enumerate(corners):
            # item  4, 2
            item = np.concatenate([item, np.ones((4, 1))], axis = 1) # 4, 3 homogeneous coors
            translate_item = item.dot(Mat.T) # 4, 2

            left_x = np.min(translate_item[:, 0])
            left_y = np.min(translate_item[:, 1])

            right_x = np.max(translate_item[:, 0])
            right_y = np.max(translate_item[:, 1])
            translated_label = [(left_x + right_x) / 2, (left_y + right_y) / 2, 
                                        right_x - left_x,  right_y - left_y]
            bbox[idx] = np.array(translated_label)
       
       


       

        # Normalize coordinates 0 - 1
        bbox[:, [1, 3]] /= img.shape[0]  # height
        bbox[:, [0, 2]] /= img.shape[1]  # width
        
        bbox[np.where(bbox>1.0)] = 1.0
        bbox[np.where(bbox<0.0)] = 0.0 
        
        bbox = np.around(bbox, decimals= 4)

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    bbox[:, 0] = 1 - bbox[:, 0]


            # random up-down flip
            ud_flip = True
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    bbox[:, 1] = 1 - bbox[:, 1]

        
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 2:] = torch.from_numpy(bbox)
            labels_out[:, 1] = torch.from_numpy(cls_id)

        # Convert
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1), labels_out, img_place, shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

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
    
if __name__=='__main__':
    args = get_args()
    kitti = LoadKitti(img_path = args.img_path, label_path = args.label_path, ids = '/data/cxg1/VoxelNet_pro/Data/training/val.txt',img_size = args.img_size, hyp=hyp, augment = args.augment)
    img, labels, _, _ = kitti[0]
    print(labels)

    img = img.permute(1, 2, 0).numpy()
    labels = labels.numpy()
    h, w = img.shape[:2]
    for label in labels:
        top = (int((label[2]+label[4]/2)*w), int((label[3]+label[5]/2)*h))
        down = (int((label[2]-label[4]/2)*w), int((label[3]-label[5]/2)*h))
        img = cv2.rectangle(img, top, down, (255, 0, 0))
    img = cv2.imwrite("rotation.png", img)                     