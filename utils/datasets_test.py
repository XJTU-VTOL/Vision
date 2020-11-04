from datasets import *
import numpy as np
import cv2

hyp = {
    'giou': 3.54,  # giou loss gain
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
    'degrees': 1.98 ,  # image rotation (+/- deg)
    'translate': 0.05 ,  # image translation (+/- fraction)
    'scale': 0.05 ,  # image scale (+/- gain)
    'shear': 0.641 # image shear (+/- deg)
}

output_name = 'fix.jpg'

def draw_boxes(img, labels):
    """
    Helper functions to draw boxes on images
    This follows the same data process in the I - II Dataset Architeture
    You just need to feed the `img` and `labels` returned by the I Dataset Class, the protocol is the same

    Input:
        img: input image data (in RGB order, format -- PIL, np.ndarray)
        labels: input labels data
    """
    if type(img) != np.ndarray:
        img = np.array(img)
    img = img[:, :, ::-1]
    coors = np.array([label['bbox'] for label in labels]) # x_left, y_left, width, height
    for label in coors:
        top = (int(label[0]), int(label[1]))
        down = (int(label[0] + label[2]), int(label[1]+label[3]))
        img = cv2.rectangle(cv2.UMat(img).get(), top, down, (255, 0, 0))
    return img


def coco_test():
    train='/data/cxg1/Data/train2014'
    valid='/data/cxg1/Data/val2014'
    TrainAnnoFile='/data/cxg1/Data/annotations/instances_train2014.json'
    ValAnnoFile='/data/cxg1/Data/annotations/instances_val2014.json'
    config = {
        'root': train,
        'annFile': TrainAnnoFile
    }
    coco = COCO(config)
    img, labels = coco[120]
    img = draw_boxes(img, labels)
    img = cv2.imwrite(output_name, img)

def kitti_test():
    train = {
        'name': 'KITTI',
        'img_path': '/data/cxg1/VoxelNet_pro/Data/training/image_2',
        'label_path': '/data/cxg1/VoxelNet_pro/Data/training/label_2',
        'ids': '/data/cxg1/VoxelNet_pro/Data/training/train.txt'
    }
    kitti = KITTI(train)
    img, labels = kitti[120]
    img = draw_boxes(img, labels)
    img = cv2.imwrite(output_name, img)

if __name__=='__main__':
    coco_test()