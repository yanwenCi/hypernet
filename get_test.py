import numpy as np

data=np.load('hyperp.npy')
print(data.shape)
f=open('test_odds.sh','w')
f.write('#!/bin/bash\n')
cmd=[]
for i in range(len(data)):
    a,b,c,d=data[i]
    cmd.append('python test_hyper.py --img-list data/data_lesion/ --model-dir checkpoints/odds_bias_gen  --mod 2 --load-weights 232 --gpu 1 --hyper_val {},{},{},{}   --pred-dir Pred_dir/results_odds >checkpoints/odds_bias_gen/test{}_log \n'.format(a,b,c,d,i))

f.writelines(cmd)
f.close()
    
