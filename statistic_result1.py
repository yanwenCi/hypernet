import os
import numpy as np
import csv

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dir',  type=str,
                    help='an integer for the accumulator')
parser.add_argument('phase',type=str)
args = parser.parse_args()

f=args.dir
model_dirs= [m for m in os.listdir(os.path.join('checkpoints',f)) if 'test_log' in m]
new=[]
for i,model_dir in enumerate(model_dirs):
    no=i
    y=''.join([j for j in model_dir if j.isdigit()])
    path_log=os.path.join('checkpoints',f,'{}_log{}'.format(args.phase,y)) # for multi models
    #path_log=os.path.join('checkpoints',f,model_dirs[i],'test_log') #for single model
    if not os.path.exists( path_log):
        continue
    print(path_log)
    log=open(path_log,'r')
    log_=log.readlines()
    log_line=log_[1].split(' ')
    log_line=[i.replace('[','').strip(',') for i in log_line]
    log_line=list(filter(None, log_line))[1:7]
    log_line=[float(k.strip().strip('[').strip(']')) for k in log_line]
    log_line=[k for k in log_line]
#std   
    log_std=log_[0]
    log_std=log_std.split(' ')
    log_std=[i.replace('[','').strip(',') for i in log_std]
    log_std=list(filter(None, log_std))[1:7]
    log_std=[float(k.strip().strip('[').strip(']')) for k in log_std]
#lesion
    log_lesion=log_[-1]
    log_lesion=log_lesion.split(' ')
    log_lesion=[i.replace('[','').strip(',') for i in log_lesion]
    log_lesion=list(filter(None, log_lesion))[1:7]
    log_lesion=[float(k.strip().strip('[').strip(']')) for k in log_lesion]

#auc
    log_auc=log_[-2]
    log_auc=log_auc.split(' ')
    log_auc=[i.replace('[','').strip(',') for i in log_auc]
    log_auc=list(filter(None,log_auc))[1:4]
    log_auc=[float(k.strip().strip('[').strip(']')) for k in log_auc]

#lesion0.5
    log_lesion5=log_[-3]
    log_lesion5=log_lesion5.split(':')[-1].split(' ')
    log_lesion5=[i.replace('[','').strip(',') for i in log_lesion5]
    log_lesion5=list(filter(None, log_lesion5))[0:6]
    log_lesion5=[float(k.strip().strip('[').strip(']')) for k in log_lesion5]

    log_dice=['%.2f(%.2f)' % (k,l) for k, l in zip(log_line, log_std)]
    print([y]+log_line+log_std+log_lesion)
    new.append([y]+log_dice+log_lesion+log_auc+log_line+log_lesion5)

    

with open(os.path.join('checkpoints',f,'results_log_{}.csv'.format(args.phase)),'w') as file:
    writer=csv.writer(file)
    writer.writerows(new)

