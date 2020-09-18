#!/bin/bash
nohup python -u train.py --batch-size 16 --cfg yolov3-tiny.cfg --data data/my_coco.data  --device 5, 2>&1 > kitti.log &