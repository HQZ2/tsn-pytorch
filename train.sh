#!/usr/bin/env bash

export PYTHONPATH="/mnt/workspace/Code/tsn-pytorch/:$PYTHONPATH"

nohup python -u main.py activitynet RGB /mnt/workspace/activitynet_info/activitynet_train_clip_all_list200.txt  /mnt/workspace/activitynet_info/activitynet_val_list.txt --arch dpn107 --num_segments 5 -p 1 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 16 -j 0 --dropout 0.8 --snapshot_pref '/mnt/workspace/model/activitynet_allclip_kinetics600_dpn107_rgb_model/activitynet_allclip_600_dpn107' > ./train-kinetics600-dpn107-allclip.log 2>&1 &
