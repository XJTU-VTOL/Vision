from .augment import hyper

train = {
    'name': 'COCO',
    'root': '/opt/data/private/COCO2017/train2017',
    'annFile': '/opt/data/private/COCO2017/annotations/instances_train2017.json',
}

val = {
    'name': 'COCO',
    'root': '/opt/data/private/COCO2017/val2017',
    'annFile': '/opt/data/private/COCO2017/annotations/instances_val2017.json',
}

coco_train_dataset_cfg = {
    'classes': 90,
    'base': train,
    'augment': False,
    'image_size': 416,
    'hyper': hyper
}

coco_val_dataset_cfg = {
    "classes": 90,
    'base': val,
    'augment': False,
    'image_size': 416,
    'hyper': hyper
}