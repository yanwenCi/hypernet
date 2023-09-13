"""
Example script for training a HyperMorph model to tune the
regularization weight hyperparameter.

If you use this code, please cite the following:

    A Hoopes, M Hoffmann, B Fischl, J Guttag, AV Dalca.
    HyperMorph: Amortized Hyperparameter Learning for Image Registration
    IPMI: Information Processing in Medical Imaging. 2021. https://arxiv.org/abs/2101.01035

Copyright 2020 Andrew Hoopes

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import numpy as np
from datetime import datetime
import surface_distance as sd
# from tqdm.keras import TqdmCallback
# tf.compat.v1.disable_eager_execution()
from metrics import pn_rate
from evaluation import lesion_metric
from sklearn.metrics import auc
import nibabel as nib
import  glob
import voxelmorph as vxm
# disable_eager_execution()
# tf.executing_eagerly()
# tf.eagerly()
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters

# training parameters
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--model_name', type=str, default='imagesTs_predfullres')
parser.add_argument('--oversample-rate', type=float, default=1,
                    help='hyperparameter end-point over-sample rate (default 0.2)')
parser.add_argument('--hyper-val', type=str, default=None)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#print(gpu_avilable)

accuracy_func=vxm.losses.Dice(with_logits=False)
preds_file = glob.glob(f'{args.model_name}/*.nii.gz')
preds_file = list(sorted(preds_file))
print(len(preds_file))
labels_file = [i.replace(args.model_name, 'labelsTs') for  i in preds_file]#glob.glob('../../nnUNet/nnUNet_raw/Dataset005_prostate/labelsTs/*.nii.gz')
zones_file = [i.replace(args.model_name, 'zonesTs') for  i in preds_file]#glob.glob('../../nnUNet/nnUNet_raw/Dataset005_prostate/zonesTs/*.nii.gz')
    # prepare loss functions and compile model

def test_loader(label_file, pred_file, zone_file):
    patient = label_file.split('/')[-1].split('_')[0]
    if patient in pred_file and patient in zone_file:
        label = nib.load(label_file).get_fdata()
        pred = nib.load(pred_file).get_fdata()
        zone = nib.load(zone_file).get_fdata()
        return label[None, ...], pred[ None,...], zone[None, ...]
    else:
        print(f'wrong files {label_file}, {pred_file}, {zone_file}')


def majority_voting(segmentation_masks):
    """
    Applies majority voting to a list of segmentation masks.

    Args:
        segmentation_masks (list of numpy arrays): List of binary segmentation masks.

    Returns:
        numpy array: Majority voted segmentation mask.
    """
    num_masks = len(segmentation_masks)
    mask_shape = segmentation_masks[0].shape
    majority_mask = np.zeros(mask_shape)

    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            for m in range(mask_shape[2]):
                votes = np.zeros(num_masks)
                for k in range(num_masks):
                    votes[k] = segmentation_masks[k][i, j, m]

                majority_vote = np.argmax(np.bincount(votes.astype(int)))
                majority_mask[i, j, m] = majority_vote

    return majority_mask

accuracy_all, lesion_all,lesion_tp_num = [], [], []
def predicting():    
    for i, (label_file, pred_file, zone_file) in enumerate(zip(labels_file, preds_file, zones_file)):
        #if i>10:
        #    break
        label, predicted1, zone = test_loader(label_file, pred_file, zone_file)
        _, predicted2, _ = test_loader(label_file, pred_file, zone_file)
        _, predicted3, _ = test_loader(label_file, pred_file, zone_file)
        #predicted = (predicted-predicted.min())/(predicted.max()-predicted.min())
        predicted = majority_voting((predicted1, predicted2, predicted3))
        p_zone=np.zeros_like(label)
        p_zone[zone==1]=1
        t_zone=np.zeros_like(label)
        t_zone[zone==2]=1
        w_zone=np.zeros_like(label)
        w_zone[zone>0]=1
        # import matplotlib.pyplot as plt
        # plt.imshow(p_zone[0,:,:,46,0])
        # plt.show()
        p_lesion=label*p_zone
        t_lesion=label*t_zone
        p_predict=predicted*p_zone
        t_predict=predicted*t_zone
        w_predict=predicted*w_zone
        accuracy = accuracy_func.loss(label, w_predict)#predicted.round())
        accuracy_p = accuracy_func.loss(p_lesion, p_predict)
        accuracy_t = accuracy_func.loss(t_lesion, t_predict)
        print(accuracy)
       # dist evaluation
        surf_dist=sd.compute_surface_distances(np.array(w_predict.squeeze().round(), dtype=bool), np.array(label.squeeze(), dtype=bool), (1,1,1))
        hausd_dist=sd.compute_robust_hausdorff(surf_dist,95)
        hausd_dist=min(40 ,hausd_dist)
        surf_dist_tz=sd.compute_surface_distances(np.array(t_predict.squeeze().round(), dtype=bool), np.array(t_lesion.squeeze(), dtype=bool), (1,1,1))
        hausd_dist_tz=sd.compute_robust_hausdorff(surf_dist_tz,95)
        hausd_dist_tz=min(40,hausd_dist_tz)
        surf_dist_pz=sd.compute_surface_distances(np.array(p_predict.squeeze().round(), dtype=bool), np.array(p_lesion.squeeze(), dtype=bool), (1,1,1))
        hausd_dist_pz=sd.compute_robust_hausdorff(surf_dist_pz,95)
        hausd_dist_pz=min(40,hausd_dist_pz)
        
        # lesion level
#        thresh=[0.25]
#        overlap_pd_tz, number_tp_pd_tz, pred_lesion_tz=pn_rate(t_lesion.squeeze(),t_predict.squeeze(), thresh ,direct='pred')
#        overlap_gt_tz, number_tp_gt_tz, gt_lesion_tz=pn_rate(t_lesion.squeeze(), t_predict.squeeze(),thresh, direct='gt')
#        overlap_pd_pz, number_tp_pd_pz, pred_lesion_pz=pn_rate(p_lesion.squeeze(),p_predict.squeeze(), thresh ,direct='pred')
#        overlap_gt_pz, number_tp_gt_pz, gt_lesion_pz=pn_rate(p_lesion.squeeze(), p_predict.squeeze(),thresh, direct='gt')
#        overlap_pd, number_tp_pd, pred_lesion=pn_rate(label[0].squeeze(),predicted.round().squeeze(), thresh ,direct='pred')
#        overlap_gt, number_tp_gt, gt_lesion=pn_rate(label[0].squeeze(), predicted.round().squeeze(), thresh, direct='gt')
        
        gt_tp_lesion, pd_tp_lesion, gt_lesion, pred_lesion=lesion_metric(w_predict.squeeze(),label[0].squeeze(),t_zone.squeeze(),p_zone.squeeze())

        if np.sum(p_lesion)<27:
            accuracy_p=np.nan
            hasud_dist_pz=np.nan
        if np.sum(t_lesion)<27:
            accuracy_t=np.nan
            hausd_dist_tz=np.nan
        accuracy_all.append([accuracy, accuracy_t, accuracy_p,hausd_dist,hausd_dist_tz,hausd_dist_pz])
        #lesion_all.append([gt_lesion[0], pred_lesion[0], gt_lesion[9],pred_lesion_tz[9],gt_lesion[18],pred_lesion[18]])
        lesion_all.append(np.hstack((gt_lesion,pred_lesion)))
        lesion_tp_num.append(np.hstack((gt_tp_lesion,pd_tp_lesion)))
        #lesion_tp_num.append([number_tp_gt,number_tp_pd, number_tp_gt_tz,number_tp_pd_tz,number_tp_gt_pz,number_tp_pd_pz])
        #print('  ',name[0], accuracy, accuracy_t, accuracy_p)
        #print(' ' ,name[0], gt_lesion_tz,pred_lesion_tz,gt_lesion_pz,pred_lesion_pz)
    sum_accu = np.array(accuracy_all).sum(axis=0)                                     
    print('std: ',np.round(np.nanstd(np.array(accuracy_all), axis=0),4).tolist())
    print('mean: ',np.round(np.nanmean(np.array(accuracy_all),axis=0),4).tolist())
    print('lesion: ', np.sum(np.array(lesion_tp_num), axis=0)/np.sum(np.array(lesion_all),axis=0).tolist())
    #print('lesion count: ', np.sum(np.array(lesion_tp_num), axis=0),np.sum(np.array(lesion_all),axis=0).tolist())
    acc_tp=np.sum(np.array(lesion_tp_num), axis=0)
    acc_lesion=np.sum(np.array(lesion_all), axis=0)
    recall,prec=acc_tp[:27]/acc_lesion[:27],acc_tp[27:]/acc_lesion[27:]
    prec[np.isnan(prec)]=0
    print('lesion0.5: ',recall[4],prec[4],recall[13],prec[13],recall[22],prec[22])
    auc_values=[]
    for sp in range(3):
        recall_,precision_=zip(*sorted(zip(recall[9*sp:9*(sp+1)].tolist(),prec[9*sp:9*(sp+1)].tolist())))
        recall_=[0]+list(recall_)+[1]
        precision_=[1]+list(precision_)+[0]
        auc_values.append(auc(np.array(recall_),np.array(precision_)))
    print('auc: ',auc_values) 
    crit1=[0.8]*9+[0.8]*9+[0.8]*9
    crit2=[0.8]*9+[0.8]*9+[0.8]*9
    res1=(np.array(recall)-np.array(crit1))**2
    res2=(np.array(prec)-np.array(crit2))**2
    res1=res1.reshape(3,9)
    res2=res2.reshape(3,9)
    find_prec,find_reca=[],[]
    for k in range(3):
        min1=np.median(np.where(res1[k,:]==np.amin(res1[k,:]))).astype(int)
        min2=np.median(np.where(res2[k,:]==np.amin(res2[k,:]))).astype(int)
        find_prec.append(prec[k*9+min1])
        find_reca.append(recall[k*9+min2])
    print('lesion: ', find_reca[0],find_prec[0],find_reca[1], find_prec[1], find_reca[2], find_prec[2])
    
if __name__=="__main__":
    predicting()
    #print('lesion: ', np.sum(np.array(lesion_tp_num), axis=0)/np.sum(np.array(lesion_all),axis=0).tolist())
    #print('std: ',np.round(np.nanstd((np.array(lesion_tp_num)/np.array(lesion_all)), axis=0),4).tolist())
    #print(sum_accu[0] / len(accuracy_all), sum_accu[1] / (len(accuracy_all) - number_t), sum_accu[2] / (len(accuracy_all) - number_p), sum_accu[3]/len(accuracy_all), sum_accu[4]/(len(accuracy_all)),sum_accu[5]/(len(accuracy_all)))
