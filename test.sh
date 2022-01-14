 #!/usr/bin/sh
 python train_hypermorph.py --img-list ./data/data_lesion  --model-dir checkpoints/odds_bias  --mod 2 --gpu 1 --lr 1e-5 --load-weights 45
