import os
from os.path import join

pwd=os.getcwd()
    
path='/raid/candi/Wen/ProstateSeg/Data/Data_by_modality_1k'

mods=['t2w','lesion','zonal_mask','dwi','adc']
split=['train', 'test', 'validation']
line=[]
f=[]
count=0
adc_lost=0
dwi_lost=0
for g, sp in enumerate(split):
    fpath='./data/data1k_cross3/'+sp
    if not os.path.exists(fpath):
            os.makedirs(fpath, mode=0o777)
    f.append(open(join(fpath,'pair_path_list.txt'), 'w'))

list_img=os.listdir(join(path, mods[0]))
list_les=os.listdir(join(path,mods[1]))
list_msk=os.listdir(join(path,mods[2]))
list_dwi=os.listdir(join(path,mods[3]))
list_adc=os.listdir(join(path,mods[4]))
for img_name in list_img:
    img_name_pref=img_name.split('.')[0]
    print(img_name_pref)
    dwi_names=[i for i in list_dwi if img_name_pref in i]
    adc_names=[i for i in list_adc if img_name_pref in i]
    print(adc_names)
    execu=len(dwi_names)*len(adc_names)
    line=''
    if execu:
        for adc_name in adc_names:
            
            for dwi_name in dwi_names:
                count+=1
                line=line+join(path, mods[0], img_name)+' '+join(path, mods[1], img_name)+' '+join(path, mods[2], img_name)\
                    +' '+join(path, mods[4], adc_name)+' '+join(path,mods[3],dwi_name)+'\n'
    else:
        if len(adc_names)!=0 and len(dwi_names)==0:
           dwi_lost+=1 
           for adc_name in adc_names:
                count+=1
                dwi_name='empty_volume.nii.gz'
                line=line+join(path, mods[0], img_name)+' '+join(path, mods[1], img_name)+' '+join(path, mods[2], img_name)\
                      +' '+join(path, mods[4], adc_name)+' '+join(pwd,dwi_name)+'\n'
        elif len(adc_names)==0 and len(dwi_names)!=0:
            adc_lost+=1
            for dwi_name in dwi_names:
                count+=1
                adc_name='empty_volume.nii.gz'
                line=line+join(path, mods[0], img_name)+' '+join(path, mods[1],img_name)+' '+join(path, mods[2], img_name)\
                        +' '+join(pwd, adc_name)+' '+join(path,mods[3],dwi_name)+'\n'
    if count%6==3:
        f[1].write(line)
    elif count%6==4:
        f[2].write(line)
    else:
        f[0].write(line)
print(count, adc_lost, dwi_lost)
f[0].close()
f[1].close()
f[2].close()
                
                

