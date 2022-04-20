#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion_hb_cross1  --model-dir checkpoints/odds_cross1_noreg_nega  --mod 2 --gpu 1 --lr 1e-6 --batch-size 6 --steps-per-epoch 120 --nega_bias True 
