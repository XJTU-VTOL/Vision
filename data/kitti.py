from data.augment import hyper

train = {
    'name': 'KITTI',
    'img_path': '/data/cxg1/VoxelNet_pro/Data/training/image_2',
    'label_path': '/data/cxg1/VoxelNet_pro/Data/training/label_2',
    'ids': '/data/cxg1/VoxelNet_pro/Data/training/train.txt'
}

val = {
    'name': 'KITTI',
    'img_path': '/data/cxg1/VoxelNet_pro/Data/training/image_2',
    'label_path': '/data/cxg1/VoxelNet_pro/Data/training/label_2',
    'ids': '/data/cxg1/VoxelNet_pro/Data/training/val.txt'
}

kitti_train_dataset_cfg = {
    'classes': 8,
    'base': train,
    'augment': False,
    'image_size': [192, 640],
    'hyper': hyper
}

kitti_val_dataset_cfg = {
    "classes": 8,
    'base': val,
    'augment': False,
    'image_size': [192, 640],
    'hyper': hyper
}