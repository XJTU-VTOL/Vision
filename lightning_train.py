from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from lightning_model import YoloLight
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from utils.datasets import LoadImagesAndLabels, LoadCOCO
import math

def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options

hyp = {
        'giou': 3.54,  # giou loss gain
        'cls': 37.4,  # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'iou_t': 0.20,  # iou training threshold
        'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
        'lrf': 0.0005,  # final learning rate (with cos scheduler)
        'lr_decay': 0.1,
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # optimizer weight decay
        'decay_epoch_list': [0.4, 0.8],
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
        'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
        'degrees': 15,  # image rotation (+/- deg)
        'translate': 0.05,  # image translation (+/- fraction)
        'scale': 0.05,  # image scale (+/- gain)
        'shear': 0.641,
        'conf_thres': 0.1, 
        'iou_thres': 0.7, 
        'multi_label': True
    }  # image shear (+/- deg)

if __name__=='__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--num_workers', type=int, default=1, help='num workers for dataloader')
    parser.add_argument('--data', type=str, default='data/my_coco.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--ckpt', default=None, type=str, help='checkpoint resume from.')
    opt = parser.parse_args()

    print(opt)

    cfg = opt.cfg
    data = opt.data
    epochs = opt.max_epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = (416, 416, 416)  # img sizes (min, max, test)

    # Image Sizes
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    img_size = imgsz_max  # initialize with max size

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    val_path = data_dict['valid']
    train_anno = data_dict['TrainAnnoFile']
    val_anno = data_dict['ValAnnoFile']

    nc = int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 90  # update coco-tuned hyp['cls'] to current dataset

    train_dataset =LoadCOCO(train_path, train_anno, img_size=img_size,
                            augment=True,
                            hyper=hyp,  # augmentation hyperparameters
                            )

    # train_dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
    #                               augment=True,
    #                               hyp=hyp,  # augmentation hyperparameters
    #                               rect=False,  # rectangular training
    #                               cache_images=False,
    #                               single_cls=False)
    
    # batch = train_dataset[4]

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=opt.num_workers,
                                                    shuffle=True,
                                                    collate_fn=train_dataset.collate_fn)

    val_dataset = LoadCOCO(val_path, val_anno, img_size=img_size, augment=False, hyper=hyp)

    # val_dataset = LoadImagesAndLabels(val_path, imgsz_test, batch_size,
    #                                     hyp=hyp,
    #                                     rect=True,
    #                                     cache_images=True,
    #                                     single_cls=False)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                num_workers=opt.num_workers,
                                                shuffle=False,
                                                collate_fn=val_dataset.collate_fn)

    checkpoint_callback = ModelCheckpoint(
        filepath='./output',
        save_top_k=4,
        verbose=True,
        monitor='precision',
        mode='max',
        save_weights_only=False,
        prefix='',
    )

    model = YoloLight(opt, hyp, nc)
    trainer = pl.Trainer.from_argparse_args(opt, distributed_backend='ddp', checkpoint_callback=checkpoint_callback, resume_from_checkpoint=opt.ckpt)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
