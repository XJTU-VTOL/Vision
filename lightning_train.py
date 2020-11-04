from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from lightning_model import YoloLight
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
import os
from utils.load_datasets import LoadDataset 
import math
import data as data_cfg

# hyper parameters for model
hyp = {
        'giou': 3.54,  # giou loss gain
        'cls': 37.4,  # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'iou_t': 0.20,  # iou training threshold
        'lr0': 0.001,  # initial learning rate (SGD=5E-3, Adam=5E-4)
        'lrf': 0.00001,  # final learning rate (with cos scheduler)
        'lr_decay': 0.1,
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # optimizer weight decay
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
        'conf_thres': 0.5, 
        'iou_thres': 0.6, 
        'multi_label': True
    }  # image shear (+/- deg)

if __name__=='__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--num_workers', type=int, default=1, help='num workers for dataloader')
    parser.add_argument('--data', type=str, default='COCO', help='dataset name')
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--ckpt', default=None, type=str, help='checkpoint resume from.')
    parser.add_argument('--freeze', action='store_true', help='Freeze non-output layers')
    parser.add_argument('--save_path', type=str, default='../coco_train', help='checkpoint save path')
    opt = parser.parse_args()

    print(opt)

    cfg = opt.cfg
    data = opt.data
    epochs = opt.max_epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    weights = opt.weights  # initial training weights

    train_dataset_cfg = data_cfg.dataset_cfg[data]["train"] # get dataset configuration
    val_dataset_cfg = data_cfg.dataset_cfg[data]["val"]
    hyp['train_dataset'] = train_dataset_cfg # add dataset configuration to hyper parameters for storage
    hyp['val_dataset'] = val_dataset_cfg

    assert train_dataset_cfg['image_size'] == val_dataset_cfg['image_size'], 'image size of train and val dataset must be the same!'
    img_size = train_dataset_cfg['image_size']

    # check image size
    gs = 32  # (pixels) grid size after down sampling
    if type(img_size) == int:
        image_size_x, image_size_y = img_size, img_size
    else:
        image_size_x, image_size_y = img_size
    assert math.fmod(image_size_x, gs) == 0, '--img-size %g must be a %g-multiple' % (image_size_x, gs)

    nc = int(train_dataset_cfg['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset, 80 is their dataset size

    train_dataset =LoadDataset(train_dataset_cfg)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=opt.num_workers,
                                                    shuffle=True,
                                                    collate_fn=train_dataset.collate_fn)

    print("Dataset size: ", len(train_dataloader))

    val_dataset = LoadDataset(val_dataset_cfg)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                num_workers=opt.num_workers,
                                                shuffle=False,
                                                collate_fn=val_dataset.collate_fn)

    checkpoint_callback = ModelCheckpoint(
        filepath=opt.save_path,
        save_top_k=4,
        verbose=True,
        monitor='mAp',
        mode='max',
        save_weights_only=False,
        prefix='',
    )

    model = YoloLight(opt, hyp, nc, transfer=opt.freeze)
    trainer = pl.Trainer.from_argparse_args(opt, checkpoint_callback=checkpoint_callback, resume_from_checkpoint=opt.ckpt)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    # save the latest one
    trainer.save_checkpoint(os.path.join(opt.save_path, "latest.ckpt"))
