#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion_cross1  --model-dir checkpoints/odds_bias123_cross_1_noreg  --mod 2 --gpu 0 --lr 1e-6 --batch-size 4 --steps-per-epoch 250 --hyper-val ,17.525383,17.525389,-17.738346,-8.580585 
