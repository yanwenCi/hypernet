#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion  --model-dir checkpoints/linear_bias  --mod 1 --gpu 1 --lr 1e-5 --batch-size 4 --steps-per-epoch 250 
