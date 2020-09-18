#!/bin/bash
nohup python -u lightning_train.py --batch-size 48 --cfg cfg/yolov3-tiny.cfg --data COCO --ckpt ../coco_train/epoch=129.ckpt --max_epochs 300 --gpus 3,4 --num_workers 48 2>&1 >lightn.log &