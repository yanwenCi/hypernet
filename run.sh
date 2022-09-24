#!/usr/bin/sh
nohup python train_hypermorph.py --img-list ./data/data_lesion_cross1  --model-dir checkpoints/hyper1_cross1_924 --mod 0 --gpu 2 --lr 1e-6 --batch-size 4 --steps-per-epoc 310 > hp1cr1.out 2>&1 &

