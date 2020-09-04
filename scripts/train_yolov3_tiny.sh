#!/bin/bash
nohup python -u lightning_train.py --batch-size 32 --cfg yolov3-tiny.cfg --data data/my_coco.data --max_epochs 300 --gpus 3,4 --num_workers 32 2>&1 >lightn.log &