#!/bin/bash
nohup python -u train.py --batch-size 16 --cfg yolov3-tiny3.cfg --data data/kitti.data  --device 6, 2>&1 > kitti.log &