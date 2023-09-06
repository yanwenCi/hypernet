#!/usr/bin/sh
nohup python train_hypermorph.py --img-list ./data/data_lesion_cross1  --model-dir checkpoints/hyper2_cross1 --mod 2 --gpu 1 --lr 1e-6 --batch-size 4 --steps-per-epoc 310 > hp2cr1.out 2>&1 &

