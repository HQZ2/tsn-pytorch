#!/usr/bin/env bash

export PYTHONPATH="/mnt/workspace/Code/tsn-pytorch/:$PYTHONPATH"

python test_models.py activitynet Flow /mnt/workspace/activitynet_info/activitynet_val_flow_list.txt  /mnt/workspace/model/activitynet_clip_kinetics600_dpn107_flow_model/activitynet_flow_clip_600_dpn107_flow_model_best_234.pth.tar --arch dpn107 -j 0 --dropout 0.7 --save_scores '/mnt/workspace/activitynet_info/kinetics600_clip_flow_test_best234.npz'