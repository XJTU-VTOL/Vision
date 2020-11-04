from .coco import coco_train_dataset_cfg, coco_val_dataset_cfg
from .kitti import kitti_train_dataset_cfg, kitti_val_dataset_cfg

coco_dataset = {
    "train": coco_train_dataset_cfg,
    "val": coco_val_dataset_cfg,
}

kitti_dataset = {
    "train": kitti_train_dataset_cfg,
    "val": kitti_val_dataset_cfg
}

dataset_cfg = {
    "COCO": coco_dataset,
    "KITTI": kitti_dataset
}