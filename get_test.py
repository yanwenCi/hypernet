import numpy as np

data=np.load('hyperp255.npy')
print(data.shape)
f=open('test_odds231.sh','w')
f.write('#!/bin/bash\n')
cmd=[]
for i in range(len(data)):
    a,b,c,d=data[i]
    print(np.round(a,4))
    cmd.append('python test_hyper.py --img-list data/data_lesion/ --model-dir checkpoints/odds_bias231  --mod 2 --load-weights 326  --gpu 2 --hyper_val ,{:>6f},{:>6f},{:>6f},{:>6f}   --pred-dir Pred_dir/results_odds132 >checkpoints/odds_bias132/test{}_log \n'.format(a,b,c,d,i))

f.writelines(cmd)
f.close()
    
