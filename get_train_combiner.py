import numpy as np
import os

decision=np.load('sample_test.npy')
data=np.load('hyperp_test.npy')
print(data.shape)
f=open('run_combiner2.sh','w')
f.write('#!/bin/bash\n')
cmd=[]
for i in range(len(data)):
    #i=np.random.randint(len(data))
    y=decision[i].tolist()
    y=y[4:]+y[0:4]
    no=[str(p) for p in y]
    no=int(''.join(no),2)
    a,b,c,d=data[i]
    if d>0:
        continue
    else:
        print(i, np.round(a,4))
#        if not os.path.exists('checkpoints/combiner2_cross1/{}'.format(i)):
#            continue
#        files=os.listdir('checkpoints/combiner2_cross1/{}'.format(i))
#        files=[int(i.split('.')[0]) for i in files if '.h5' in i]
#        if len(files)==0:
#            continue
#        e=max(files)
#        print(e)
        cmd.append('python train_nohyper.py --img-list data/data_lesion_cross1/ --lr 1e-4  --model-dir checkpoints/combiner2_cross1/{}  --mod 2  --gpu 1 --epoch 20 --patience 3  --batch-size 4 --steps-per-epoch 300  --hyper-val ,{:>6f},{:>6f},{:>6f},{:>6f}\n'.format(no,a,b,c,d))
        #cmd.append('python test_hyper.py --img-list data/data_lesion_cross1/ --model-dir checkpoints/combiner2_cross1/{}  --mod 2 --load-weights {}  --net combiner  --gpu 2 --hyper_val ,{:>6f},{:>6f},{:>6f},{:>6f}   --pred-dir Pred_dir/combiner2_cross1/{} >checkpoints/combiner2_cross1/{}/test_log\n'.format(no,e,a,b,c,d,no,no))

f.writelines(cmd)
f.close()
    
