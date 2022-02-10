#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion  --model-dir checkpoints/odds_bias132  --mod 2 --gpu 0 --lr 1e-5 --batch-size 9 --steps-per-epoch 100 
