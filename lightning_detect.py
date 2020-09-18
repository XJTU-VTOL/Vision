from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from lightning_model import YoloLight
from torch import nn
import pytorch_lightning as pl
from utils.datasets import letterbox
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from utils.load_datasets import LoadDataset 
import math
import importlib
from lightning_train import hyp
import numpy as np
import cv2

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3-tiny.cfg', help='*.cfg path')
    parser.add_argument('--ckpt', default='weights/yolov3-tiny.pt', type=str, help='checkpoint resume from.')
    parser.add_argument('--img', type=str, default='data/samples/bus.jpg', help='path to img')
    opt = parser.parse_args()
    print(opt)
    nc = 90
    img = cv2.imread(opt.img)
    img, _, _ = letterbox(img, auto=False)
    img = img[np.newaxis, :]

    model = YoloLight(opt, hyp, nc)
    checkpoint = torch.load(opt.ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    with torch.no_grad():
        img_out = model.detect(img)
    
    cv2.imwrite('out.jpg', img_out[0])


    
