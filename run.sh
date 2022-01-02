#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion  --model-dir checkpoints/odds_bias_fix  --mod 2 --gpu 1 --lr 5e-5 --batch-size 4 --steps-per-epoch 250 
