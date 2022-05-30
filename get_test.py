import numpy as np

data=np.load('hyperp_test.npy')
print(data.shape)
f=open('test_hyper2-2.sh','w')
decision=np.load('sample_test.npy')
f.write('#!/bin/bash\n')
cmd=[]
for i in range(len(data)):
    #i=np.random.randint(len(data))
    a,b,c,d=data[i]
    y=decision[i].tolist()
    y=y[4:]+y[0:4]
    no=[str(p) for p in y]
    no=int(''.join(no),2)
    
    if d>0:
        continue
    else:
        print(i, np.round(a,4))
        #cmd.append('python train_hypermorph.py --img-list data/data_lesion_cross1/ --model-dir checkpoints/combiner2_cross1/{}  --mod 2  --gpu 2 --epoch 20  --batch-size 4  --steps-per-epoch 300  --hyper-val ,{:>6f},{:>6f},{:>6f},{:>6f}\n'.format(i,a,b,c,d))

        cmd.append('python test_hyper.py --img-list data/data_lesion_cross2/ --model-dir checkpoints/hyper2_cross2  --mod 2 --load-weights 292  --grid-search True  --gpu 2 --hyper_val ,{:>6f},{:>6f},{:>6f},{:>6f}   --pred-dir Pred_dir/hyper2_cross2/{} >checkpoints/hyper2_cross2/valid{}_log\n'.format(a,b,c,d,no,no))

f.writelines(cmd)
f.close()
    
