#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion_cross2  --model-dir checkpoints/hyper2_cross2 --mod 2 --gpu 1 --lr 1e-5 --batch-size 4 --steps-per-epoch 300  
