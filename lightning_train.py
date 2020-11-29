from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from yaml.loader import FullLoader
from model import create_model
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from utils.load_datasets import LoadDataset 
from utils.common_utils import updated_config
import math
import data as data_cfg

if __name__=='__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path for training')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloader')
    parser.add_argument('--data', type=str, default='COCO', help='dataset name')
    parser.add_argument('--model', type=str, default="YOLO", help="model name")
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
    # configuration dictionary
    config = {}

    with open(cfg, 'r') as fs:
        config = yaml.load(fs, Loader=FullLoader)
        # log all lightning flags here
        config = updated_config(config, opt)

    train_dataset_cfg = data_cfg.dataset_cfg[data]["train"] # get dataset configuration
    val_dataset_cfg = data_cfg.dataset_cfg[data]["val"]
    config['train_dataset'] = train_dataset_cfg # add dataset configuration to hyper parameters for storage
    config['val_dataset'] = val_dataset_cfg

    assert train_dataset_cfg['image_size'] == val_dataset_cfg['image_size'], 'image size of train and val dataset must be the same!'
    img_size = train_dataset_cfg['image_size']

    # check image size for anchor-based method
    gs = config['gs'] if 'gs' in config.keys() else -1 # (pixels) grid size after down sampling
    if gs > 0:
        if type(img_size) == int:
            image_size_x, image_size_y = img_size, img_size
        else:
            image_size_x, image_size_y = img_size
        assert math.fmod(image_size_x, gs) == 0, '--img-size %g must be a %g-multiple' % (image_size_x, gs)

    nc = int(train_dataset_cfg['classes'])  # number of classes
    config['cls'] *= nc # update coco-tuned hyp['cls'] to current dataset, 80 is their dataset size
    config['nc'] = nc
 
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
        monitor='ap',
        mode='max',
        save_weights_only=False,
        prefix='',
    )

    model = create_model(config)
    trainer = pl.Trainer.from_argparse_args(opt, checkpoint_callback=checkpoint_callback, resume_from_checkpoint=opt.ckpt)

    print(opt.auto_lr_find)
    # find lr automatically if auto_lr_find is True
    if opt.auto_lr_find:
        # Run learning rate finder
        trainer.tune(model, train_dataloader, val_dataloader)

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    # save the latest one
    trainer.save_checkpoint(os.path.join(opt.save_path, "latest.ckpt"))
