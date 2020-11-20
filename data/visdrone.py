from .augment import hyper

train = {
    'name': 'VisDrone',
    'img_path': '/opt/data/private/VisDrone/VisDrone2019-DET-train/images',
    'label_path': '/opt/data/private/VisDrone/VisDrone2019-DET-train/annotations',
}

val = {
    'name': 'VisDrone',
    'img_path': '/opt/data/private/VisDrone/VisDrone2019-DET-val/images',
    'label_path': '/opt/data/private/VisDrone/VisDrone2019-DET-val/annotations',
}

VisDrone_train_dataset_cfg = {
    'classes': 4,
    'base': train,
    'augment': True,
    'image_size': [192, 640],
    'hyper': hyper
}

VisDrone_val_dataset_cfg = {
    "classes": 4,
    'base': val,
    'augment': True,
    'image_size': [192, 640],
    'hyper': hyper
}