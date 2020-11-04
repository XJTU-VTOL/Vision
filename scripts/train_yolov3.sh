#!/bin/bash
nohup python -u lightning_train.py --batch-size 16 --cfg cfg/yolov3.cfg --data VisDrone --weights weights/yolov3.pt --max_epochs 300 --save_path ../yolov3_coco --gpus 1 --num_workers 8 2>&1 >lightn.log &
