#!/bin/bash
nohup python -u lightning_train.py --batch-size 16 --cfg cfg/yolov3.cfg --data COCO --max_epochs 300 --save_path ../yolov3_coco --gpus 3,7 --num_workers 48 2>&1 >lightn.log &