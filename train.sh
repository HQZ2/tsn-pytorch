#!/usr/bin/env bash

export PYTHONPATH="/mnt/workspace/Code/tsn-pytorch/:$PYTHONPATH"

python main.py activitynet RGB /mnt/workspace/activitynet_info/activitynet_train_clip_all_list.txt  /mnt/workspace/activitynet_info/activitynet_val_list.txt --arch dpn107 --num_segments 5 -p 1 --gd 20 --lr 0.0001 --lr_steps 30 60 --epochs 40 -b 16 -j 0 --dropout 0.8 --snapshot_pref '/mnt/workspace/model/activitynet_clip_kinetics600_dpn107_rgb_model2/activitynet_clip_600_dpn107'