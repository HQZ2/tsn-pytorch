#!/usr/bin/env bash

export PYTHONPATH="/mnt/workspace/Code/tsn-pytorch/:$PYTHONPATH"

python test_models.py activitynet RGB /mnt/workspace/activitynet_info/activitynet_val_list.txt  /mnt/workspace/model/activitynet_clip_kinetics400_resnet152_rgb_model/activitynet_clip_resnet152_rgb_model_best_059.pth.tar --arch resnet152 -j 0 --dropout 0.7 --save_scores '/mnt/workspace/activitynet_info/kinetics400_clip_RGB_test.npz'