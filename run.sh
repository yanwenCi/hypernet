#!/usr/bin/sh
source ../env_py36/bin/activate
python train_hypermorph.py --img-list ./data/data_lesion  --model-dir checkpoints/linear_sum_wo1  --mod 0 --gpu 2 --lr 1e-5 --batch-size 4 --steps-per-epoch 250
