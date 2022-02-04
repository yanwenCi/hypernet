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
    log=open(os.path.join('checkpoints',f,'test{}_log'.format(i)),'r')
    log_line=log.readlines()[-1]
    log_line=log_line.split(' ')
    log_line=[float(k) for k in log_line]
    y=Y[i]
    print(log_line)
    new.append(hyp.tolist()+log_line+y.tolist())
    

with open(os.path.join('checkpoints',f,'results_log.csv'),'w') as file:
    writer=csv.writer(file)
    writer.writerows(new)

