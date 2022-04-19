#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion_cross1  --model-dir checkpoints/odds_cross1_noreg_nega  --mod 2 --gpu 2 --lr 1e-6 --batch-size 4 --steps-per-epoch 250  
