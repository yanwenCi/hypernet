import numpy as np

data=np.load('hyperp255.npy')
print(data.shape)
f=open('test_odds255.sh','w')
f.write('#!/bin/bash\n')
cmd=[]
for i in range(len(data)):
    a,b,c,d=data[i]
    print(np.round(a,4))
    cmd.append('python test_hyper.py --img-list data/data_lesion/ --model-dir checkpoints/odds_bias_gen  --mod 2 --load-weights 232 --gpu 1 --hyper_val ,{:>6f},{:>6f},{:>6f},{:>6f}   --pred-dir Pred_dir/results_odds255 >checkpoints/odds_bias255/test{}_log \n'.format(a,b,c,d,i))

f.writelines(cmd)
f.close()
    
