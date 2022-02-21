#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion_cross1  --model-dir checkpoints/odds_bias123_cross2  --mod 2 --gpu 0 --lr 5e-5 --batch-size 9 --steps-per-epoch 150 
