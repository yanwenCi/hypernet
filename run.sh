#!/usr/bin/sh
python train_hypermorph.py --img-list ./data/data_lesion_hb_cross3  --model-dir checkpoints/hyper2_hb_cross3 --mod 2 --gpu 1 --lr 1e-6 --batch-size 4 --steps-per-epoc 210
