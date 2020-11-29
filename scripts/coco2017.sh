#!/bin/bash
mkdir -p ../coco2017
nohup python -u lightning_train.py --batch-size 32 --accumulate_grad_batches 4 --cfg cfg/yolov3.cfg --data COCO --weights weights/yolov3.pt --max_epochs 300 --distributed_backend ddp --save_path ../coco2017 --gpus 0,1 --num_workers 16 2>&1 >lightn.log &

# mkdir -p ../coco2017_freeze
# nohup python -u lightning_train.py --batch-size 32 --accumulate_grad_batches 4 --cfg cfg/yolov3.cfg --data COCO --weights weights/yolov3.pt --max_epochs 300 --distributed_backend ddp --save_path ../coco2017_freeze --gpus 0,1 --num_workers 16 2>&1 >lightn.log &