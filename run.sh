#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion_cross1  --model-dir checkpoints/hyper2_cross1_wg --mod 2 --gpu 1 --lr 1e-6 --batch-size 4 --steps-per-epoch 300  --hyper-val ,18,18,-0.2,-8.5
