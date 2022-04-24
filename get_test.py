import numpy as np

data=np.load('hyperp_test.npy')
print(data.shape)
f=open('run_combiner2.sh','w')
f.write('#!/bin/bash\n')
cmd=[]
for j in range(len(data)):
    i=np.random.randint(len(data))
    a,b,c,d=data[i]
    if d>1:
        continue
    else:
        print(i, np.round(a,4))
        cmd.append('python train_hypermorph.py --img-list data/data_lesion_cross1/ --model-dir checkpoints/combiner2_cross1/{}  --mod 2  --gpu 2 --epoch 20  --batch-size 4  --steps-per-epoch 300  --hyper-val ,{:>6f},{:>6f},{:>6f},{:>6f}\n'.format(i,a,b,c,d))
    #cmd.append('python test_hyper.py --img-list data/data_lesion_cross1/ --model-dir checkpoints/odds_noreg_cross1  --mod 2 --load-weights 300  --gpu 0 --hyper_val ,{:>6f},{:>6f},{:>6f},{:>6f}   --pred-dir Pred_dir/odds_noreg/{} >checkpoints/odds_noreg_cross1/test{}_log\n'.format(a,b,c,d,i,i))

f.writelines(cmd)
f.close()
    
