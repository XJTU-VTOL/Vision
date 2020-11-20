#!/bin/bash
# train with lr-find
# mkdir -p ../lr_find_no_freeze
# nohup python -u lightning_train.py --auto_lr_find True --batch-size 32 --distributed_backend ddp --accumulate_grad_batches 4 --cfg cfg/yolov3.cfg --data VisDrone --weights weights/yolov3.pt --max_epochs 300 --save_path ../lr_find_no_freeze --gpus 0,1 --num_workers 16 2>&1 >lightn.log &

# train with manually set auto_lr_find
mkdir -p ../no_lr_find_no_freeze
nohup python -u lightning_train.py --auto_lr_find False --batch-size 48 --accumulate_grad_batches 2 --cfg cfg/yolov3.cfg --data VisDrone --weights weights/yolov3.pt --max_epochs 300 --save_path ../no_lr_find_no_freeze --distributed_backend ddp --gpus 0,1 --num_workers 16 2>&1 >lightn.log &

# train with lr-find and freeze layers
# mkdir -p ../lr_find_freeze
# nohup python -u lightning_train.py --auto_lr_find True --batch-size 32 --accumulate_grad_batches 4 --cfg cfg/yolov3.cfg --data VisDrone --weights weights/yolov3.pt --max_epochs 300 --freeze --save_path ../lr_find_freeze --gpus 0,1 --num_workers 16 2>&1 >lightn.log &

# train without lr-find and freeze layers
# mkdir -p ../no_lr_find_freeze
# nohup python -u lightning_train.py --auto_lr_find False --batch-size 32 --accumulate_grad_batches 4 --cfg cfg/yolov3.cfg --data VisDrone --weights weights/yolov3.pt --max_epochs 300 --freeze --save_path ../yolov3_coco --gpus 0,1 --num_workers 16 2>&1 >lightn.log &
