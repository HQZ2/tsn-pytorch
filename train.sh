#!/usr/bin/env bash

export PYTHONPATH="/mnt/workspace/Code/tsn-pytorch/:$PYTHONPATH"

python main.py activitynet RGB /mnt/workspace/activitynet_info/activitynet_train_clip_list.txt  /mnt/workspace/activitynet_info/activitynet_val_list.txt --arch resnet152 --num_segments 3 -p 1 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 64 -j 0 --dropout 0.8 --snapshot_pref '/mnt/workspace/model/activitynet_clip_kinetics400_resnet152_rgb_model/activitynet_clip_resnet152'