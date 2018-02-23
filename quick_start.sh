#!/bin/bash
# arguments
## $1: GPU ID
# ------------------------------------------------------
# steps
## 0. download a pre-trained VGG-16 model and save as save/vgg16.npy
## 1. train with sparsity and monotonicity constraints, prune network, and save a sparse parameter dictionary in save_dir
## 2. aggregate losses at different fidelity levels and then fine-tune the pruned model
## 3. evaluation the performance of different fidelity levels
wget -O save/vgg16.npy "https://www.dropbox.com/s/6u5jp5wzlwajxu1/vgg16.npy?dl=0"
CUDA_VISIBLE_DEVICES=$1 python3 train.py --prof_type linear --lambda_s 0.001 --lambda_m 0.001 --decay 0.00005 --keep_prob 1 --save_dir save/ --init_from save/vgg16.npy
CUDA_VISIBLE_DEVICES=$1 python3 train.py --tesla 1 --decay 0.00005 --keep_prob 1 --init_from save/sparse_dict.npy --save_dir save/
CUDA_VISIBLE_DEVICES=$1 python3 test.py --keep_prob 1 --init_from save/finetune_dict.npy --output save/output.csv
