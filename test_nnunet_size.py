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
from metrics import pn_rate, lesion_size
from evaluation import lesion_metric
from sklearn.metrics import auc
import nibabel as nib
import  glob
import voxelmorph as vxm
from scipy.stats import mannwhitneyu
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
preds_file = glob.glob(f'../nnUnet/nnUNet_raw/Dataset005_prostate/{args.model_name}/*.nii.gz')
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
        return label, pred, zone
    else:
        print(f'wrong files {label_file}, {pred_file}, {zone_file}')
        
        
size_gt, size_pred= [], []
def predicting():    
    size_gt, size_pred, dice= [], [], []
    for i, (label_file, pred_file, zone_file) in enumerate(zip(labels_file, preds_file, zones_file)):
        #if i>10:
        #    break
        label, predicted, zone = test_loader(label_file, pred_file, zone_file)
        #predicted = (predicted-predicted.min())/(predicted.max()-predicted.min())
        #dice.append(2*np.sum(label*predicted)/(np.sum(label)+np.sum(predicted)))       
        size_pred_patient, size_gt_patient, dice_patient = lesion_size(label, predicted)

        size_gt+=size_gt_patient
        size_pred+=size_pred_patient
        dice+=dice_patient
    size_gt = np.array(size_gt)
    size_pred = np.array(size_pred)
    mean_gt = np.mean(size_gt)
    dice_big=np.array(dice)[size_gt>4000]
    dice_small=np.array(dice)[size_gt<4000.1]
    print(len(dice_big), len(dice_small))
    statistic, p_value = mannwhitneyu(dice_big, dice_small)
    print("Mann-Whitney U statistic:", statistic)
    print("P-value:", p_value)
    print(mean_gt, 'big, small', np.mean(np.array(dice_big)), np.mean(np.array(dice_small)))

if __name__=="__main__":
    predicting()
    #print('lesion: ', np.sum(np.array(lesion_tp_num), axis=0)/np.sum(np.array(lesion_all),axis=0).tolist())
    #print('std: ',np.round(np.nanstd((np.array(lesion_tp_num)/np.array(lesion_all)), axis=0),4).tolist())
    #print(sum_accu[0] / len(accuracy_all), sum_accu[1] / (len(accuracy_all) - number_t), sum_accu[2] / (len(accuracy_all) - number_p), sum_accu[3]/len(accuracy_all), sum_accu[4]/(len(accuracy_all)),sum_accu[5]/(len(accuracy_all)))
