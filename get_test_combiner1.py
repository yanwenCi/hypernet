import numpy as np
import os
import re

f=open('test_combiner1.sh','w')
cmd=[]
path=r'checkpoints/combiner1_cross1'
print(os.listdir(path))
file_list=[l for l in os.listdir(path) if 'mix' in l]
for i in range(len(file_list)):
    #i=np.random.randint(len(data))
    no=re.findall(r'\d+',file_list[i])[0]
    if len(no)==3:
        a,b,c=int(no[0])/10,int(no[1])/10,int(no[2])/10
    else:
        if no=='1000':
            a,b,c=1,0,0
        elif no=='0010':
            a,b,c=0,0,1
        elif no=='0100':
            a,b,c=0,1,0
    # print(i, np.round(a,4))
    if not os.path.exists(os.path.join(path,file_list[i])):
        continue
    files=os.listdir(os.path.join(path, file_list[i]))
    files=[int(i.split('.')[0]) for i in files if '.h5' in i]
    files.sort()
    if len(files)==0:
        continue
    if max(files)==20:
        e=files[-2]
        print(e)
    else:
        e=max(files)
        #cmd.append('python train_nohyper.py --img-list data/data_lesion_cross1/ --lr 1e-4  --model-dir checkpoints/combiner2_cross1/{}  --mod 2  --gpu 1 --epoch 20 --patience 3  --batch-size 4 --steps-per-epoch 300  --hyper-val ,{:>6f},{:>6f},{:>6f},{:>6f}\n'.format(no,a,b,c,d))
    cmd.append('python test_hyper.py --img-list data/data_lesion_cross1/ --model-dir '+os.path.join(path,file_list[i] )+'  --mod 0 --load-weights {}  --net combiner  --gpu 2 --hyper_val ,{:>2f},{:>2f},{:>2f}   --pred-dir Pred_dir/combiner2_cross1/mix{} >'.format(e,a,b,c,no)+os.path.join(path,file_list[i])+'/test_log\n')

f.writelines(cmd)
f.close()
    
