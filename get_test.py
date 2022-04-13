import numpy as np

data=np.load('hyperp_test.npy')
print(data.shape)
f=open('test_odds123.sh','w')
f.write('#!/bin/bash\n')
cmd=[]
for i in range(len(data)):
    a,b,c,d=data[i]
    if d>1:
        continue
    else:
        print(np.round(a,4))
        cmd.append('python test_hyper.py --img-list data/data_lesion_cross1/ --model-dir checkpoints/odds_valid_cross1  --mod 2 --load-weights 269  --gpu 0 --hyper_val ,{:>6f},{:>6f},{:>6f},{:>6f}   --pred-dir Pred_dir/odds_valid/{} >checkpoints/odds_valid_cross1/test{}_log\n'.format(a,b,c,d,i,i))

f.writelines(cmd)
f.close()
    
