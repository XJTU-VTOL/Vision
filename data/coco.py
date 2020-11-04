from .augment import hyper

train = {
    'name': 'COCO',
    'root': '/opt/data/private/COCO/train2014',
    'annFile': '/opt/data/private/COCO/annotations/instances_train2014.json',
}

val = {
    'name': 'COCO',
    'root': '/opt/data/private/COCO/val2014',
    'annFile': '/opt/data/private/COCO/annotations/instances_val2014.json',
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