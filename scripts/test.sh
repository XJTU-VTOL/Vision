#!/bin/bash
python lightning_train.py --batch-size 16 --cfg cfg/yolov3.cfg  --fast_dev_run True --data VisDrone --gpus 0 --num_workers 16 