import os
import numpy as np
import csv

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dir',  type=str,
                    help='an integer for the accumulator')
parser.add_argument('npy', type=str)
parser.add_argument('Y', type=str)
args = parser.parse_args()
npy_file=args.npy

f=args.dir
data=np.load(npy_file)
Y=np.load(args.Y)
new=[]
for i in range(len(data)):
    hyp=data[i]
    if hyp[-1]>1:
        continue
    log=open(os.path.join('checkpoints',f,'test{}_log'.format(i)),'r')
    log_=log.readlines()
    log_line=log_[1].split(' ')
    log_line=[i.replace('[','').strip(',') for i in log_line]
    log_line=list(filter(None, log_line))[1:7]
    log_line=[float(k.strip().strip('[').strip(']')) for k in log_line]
   
    log_std=log_[0]
    log_std=log_std.split(' ')
    log_std=[i.replace('[','').strip(',') for i in log_std]
    log_std=list(filter(None, log_std))[1:7]
    log_std=[float(k.strip().strip('[').strip(']')) for k in log_std]

    y=Y[i].tolist()
    y=y[4:]+y[0:4]
    no=[str(p) for p in y]
    no=int(''.join(no),2)
    print(log_line,y,no)
    new.append([no]+y+hyp.tolist()+log_line+log_std)
    

with open(os.path.join('checkpoints',f,'results_log.csv'),'w') as file:
    writer=csv.writer(file)
    writer.writerows(new)

