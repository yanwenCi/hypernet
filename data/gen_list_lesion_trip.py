import numpy as np
import os
from os.path import join

path=r'/raid/candi/Wen/ProstateSeg/Data/Data_by_modality_multib'

mods=['t2w','lesion','zonal_mask','dwi','adc']
split=['train', 'test', 'valid','whole_set']
f=[]
pairs_num=0
count=0
adc_num_list,dwi_num_list=[],[]
for g, sp in enumerate(split):
    fpath='/raid/candi/Wen/ProstateSeg/hypernet/data/data_lesion_hb_cross2/'+sp
    if not os.path.exists(fpath):
            os.makedirs(fpath, mode=0o777)
    f.append(open(join(fpath,'pair_path_list.txt'), 'w'))

list_img=os.listdir(join(path, mods[0]))
list_les=os.listdir(join(path,mods[1]))
list_msk=os.listdir(join(path,mods[2]))
list_dwi=os.listdir(join(path,mods[3]))
list_adc=os.listdir(join(path,mods[4]))
dwi_num, adc_num,both=0,0,0
for img_name in sorted(list_img):
    img_name_pref=img_name.split('.')[0]
    #print(img_name_pref)
    dwi_names_=sorted([i for i in list_dwi if img_name_pref in i])
    dwi_names=[i for i in dwi_names_ if 'dwi1k4'  in i]+[i for i in dwi_names_ if 'dwi2k' in i]+[i for i in dwi_names_ if 'dwi1k5' in i]
    if len(dwi_names)==0:
        dwi_names=[i for i in dwi_names_ if 'dwi1k' in i]
    if len(dwi_names)==0:
        dwi_names=[i for i in dwi_names_ if 'dwi8h' in i]
    #print(dwi_names)
    if len(dwi_names)>3:
        print(img_name,'dwi num', len(dwi_names))
    adc_names=[i for i in list_adc if img_name_pref in i]
    if len(adc_names)>3:
        print(img_name, 'adc_num', len(adc_names))

    if len(adc_names):
        adc_num+=1 
        adc_num_list.append(len(adc_names))
    if len(dwi_names):
        dwi_num_list.append(len(dwi_names))
        dwi_num+=1
    execu=len(dwi_names)*len(adc_names)
    if execu:
        line=''
        for adc_name in adc_names:            
            for dwi_name in dwi_names:
#                dwi1k_names=[i for i in dwi_names if 'dwi1k' in i]
#                if not dwi1k_names:
#                    dwi_only=list(set(dwi_names)-set(dwi1k_names))
            #for dwi1k_name in dwi1k_names:
                pairs_num+=1
                line=line+join(path, mods[0], img_name)+' '+join(path, mods[1], img_name)+' '+join(path, mods[2], img_name)\
                    +' '+join(path, mods[4], adc_name)+' '+join(path,mods[3],dwi_name)+'\n'
        count+=1

    if count%6==1:
        f[1].write(line)
    elif count%6==2:
        f[2].write(line)
    else:
        f[0].write(line)
f[3].write(line)
jj=[i for i in np.unique(np.array(adc_num_list)).tolist()]
JJ=[adc_num_list.count(float(j)) for j in jj]
print('adc',jj,JJ)
jj=[i for i in np.unique(np.array(dwi_num_list))]
JJ=[dwi_num_list.count(float(j)) for j in jj]
print('dwi', jj, JJ)
print('patient {}, pairs {}, adc {}, dwi {}, both {}'.format(len(list_img), pairs_num, adc_num, dwi_num, count))
f[0].close()
f[1].close()
f[2].close()
                
                

