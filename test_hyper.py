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
import tensorflow as tf
import voxelmorph as vxm
from tensorflow.keras import backend as K
from datetime import datetime
import surface_distance as sd
# from tqdm.keras import TqdmCallback
# tf.compat.v1.disable_eager_execution()
from metrics import pn_rate
from tensorflow.python.framework.ops import disable_eager_execution
from evaluation import lesion_metric
from sklearn.metrics import auc
# disable_eager_execution()
# tf.executing_eagerly()
# tf.eagerly()
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--hyper_gen', help='atlas filename')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--pred-dir', default='Pred_dir')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--test-reg', nargs=3,
                    help='example registration pair and result (moving fixed moved) to test')

# training parameters
parser.add_argument('--gpu', default='2', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=600,
                    help='number of training epochs (default: 6000)')
parser.add_argument('--steps-per-epoch', type=int, default=500,
                    help='steps per epoch (default: 100)')
parser.add_argument('--load-weights', required = True, help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 1e-4)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--grid-search', type=bool, default=True)
parser.add_argument('--net', default='hyper', choices=['hyper','combiner'])

# loss hyperparameters
parser.add_argument('--mod', type=int, default=None)
parser.add_argument('--activation', type=str, default=None)
parser.add_argument('--hyper_num', type=int, default=3)

parser.add_argument('--image-loss', default='dice',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--image-sigma', type=float, default=0.05,
                    help='estimated image noise for mse image scaling (default: 0.05)')
parser.add_argument('--oversample-rate', type=float, default=1,
                    help='hyperparameter end-point over-sample rate (default 0.2)')
parser.add_argument('--hyper-val', type=str, default=None)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpu_avilable = tf.config.experimental.list_physical_devices('GPU')
#print(gpu_avilable)

def random_hyperparam(hyper_num):
    if args.hyper_val is not None:
        hyper_val = np.array([float(i) for i in args.hyper_val.split(',')[1:]])
    else:
        if args.mod == 2:
            #hyper_val = hyperps[60]
            #hyper_val = np.random.uniform(low=0, high=1, size=(hyper_num,))
            #hyper_val = np.random.uniform(low=-20, high=20, size=(hyper_num,))
            hyper_val = hyperps[np.random.randint(0, len(hyperps)*args.oversample_rate)]
        else:
            hyper_val =np.random.dirichlet(np.ones(3), size=1)[0]
    return hyper_val

logdir = args.model_dir + "/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel
# scan-to-scan generator
if args.grid_search:
    __phase='valid'
else:
    __phase='test'
base_generator = vxm.generators.multi_mods_gen(
    args.img_list, phase=__phase, batch_size=args.batch_size, test= True, add_feat_axis=add_feat_axis)
# random hyperparameter generator

hyperps = np.load('hyperp_test.npy')


if args.mod ==2 :
    # weighted 0
    # logic 1
    args.hyper_num += 1
    args.activation=None
elif args.mod==0:
    args.activation='sigmoid'
#elif args.mod==3:
    

# extract shape and number of features from sampled input
sample_shape = next(base_generator)[0][0].shape
inshape = sample_shape[1:-1]
nfeats = sample_shape[-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 16]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')
lesion_all=[]
lesion_tp_num=[]
accuracy_all=[]
# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
save_file = os.path.join(args.pred_dir, args.model_dir.split('/')[-1])
if not os.path.exists(save_file):
    os.makedirs(save_file)
results=[]
with tf.device(device):
    # build the model
    if args.net=='combiner':
        model = vxm.networks.UnetDense(
                     inshape=inshape,
                     nb_unet_features=[enc_nf, dec_nf],
                     src_feats=nfeats,
                     trg_feats=nfeats,
                     unet_half_res=False,
                    activate=args.activation,
                    nb_hyp_params=args.hyper_num)
    elif args.net=='hyper':
        model = vxm.networks.HyperUnetDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        src_feats=nfeats,
        trg_feats=nfeats,
        unet_half_res=False,
        nb_hyp_params=args.hyper_num,
        activation=args.activation)
#    #print(model.summary())
#    # load initial weights (if provided)

    model.load_weights(os.path.join(model_dir, '{:04d}.h5'.format(int(args.load_weights))))
    #print('loading weights from {:04d}.h5'.format(int(args.load_weights)))

    # prepare image loss
    hyper_val = model.references.hyper_val
    if args.mod>=2:
        accuracy_func=vxm.losses.Dice(with_logits=False)
    else:
        accuracy_func = vxm.losses.Dice(with_logits=False)
    number_t,number_p=0,0
    number=[]
    # prepare loss functions and compile model
    for i, data in enumerate(base_generator):
        #if i>10:
        #    break
        hyper_val = random_hyperparam(args.hyper_num)
        hyp = np.array([hyper_val for _ in range(args.batch_size)])
        inputs, outputs, zone, name = data
        inputs = (*inputs, hyp)
        predicted = model.predict(inputs)
        if isinstance(predicted,list):
            predicted=predicted[0]
            #predicted = (predicted-predicted.min())/(predicted.max()-predicted.min())
        #print(predicted.max())
        p_zone=np.zeros_like(outputs[0])
        p_zone[zone==1]=1
        t_zone=np.zeros_like(outputs[0])
        t_zone[zone==2]=1
        w_zone=np.zeros_like(outputs[0])
        w_zone[zone>0]=1
        # import matplotlib.pyplot as plt
        # plt.imshow(p_zone[0,:,:,46,0])
        # plt.show()
        p_lesion=outputs[0]*p_zone
        t_lesion=outputs[0]*t_zone
        p_predict=predicted*p_zone
        t_predict=predicted*t_zone
        w_predict=predicted*w_zone
        accuracy = accuracy_func.loss(outputs[0], w_predict)#predicted.round())
        accuracy_p = accuracy_func.loss(p_lesion, p_predict)
        accuracy_t = accuracy_func.loss(t_lesion, t_predict)
        print(accuracy)
       # dist evaluation
        surf_dist=sd.compute_surface_distances(np.array(w_predict.squeeze().round(), dtype=bool), np.array(outputs[0].squeeze(), dtype=bool), (1,1,1))
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
#        overlap_pd, number_tp_pd, pred_lesion=pn_rate(outputs[0].squeeze(),predicted.round().squeeze(), thresh ,direct='pred')
#        overlap_gt, number_tp_gt, gt_lesion=pn_rate(outputs[0].squeeze(), predicted.round().squeeze(), thresh, direct='gt')
        
        gt_tp_lesion, pd_tp_lesion, gt_lesion, pred_lesion=lesion_metric(w_predict.squeeze(),outputs[0].squeeze(),t_zone.squeeze(),p_zone.squeeze())

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
        if i%100==0:
            seg_result = predicted.squeeze()
            #print('%d-th mean accuracy: %f' % (i, np.array(accuracy_all).mean(axis=0)))
            vxm.py.utils.save_volfile(seg_result, os.path.join(save_file, '%s_dice%d_%.4f.nii.gz' % (name[0].split('.')[0],i, accuracy)))        
            vxm.py.utils.save_volfile(inputs[0].squeeze(), os.path.join(save_file, '%s_dice%d_%.4f_t2w.nii.gz' % (name[0].split('.')[0],i, accuracy)))
            vxm.py.utils.save_volfile(inputs[2].squeeze(), os.path.join(save_file, '%s_dice%d_%.4f_adc.nii.gz' % (name[0].split('.')[0],i, accuracy))) 
            vxm.py.utils.save_volfile(inputs[1].squeeze(), os.path.join(save_file, '%s_dice%d_%.4f_dwi.nii.gz' % (name[0].split('.')[0], i,accuracy)))
            vxm.py.utils.save_volfile(outputs[0].squeeze(), os.path.join(save_file, '%s_dice%d_%.4f_label.nii.gz'% (name[0].split('.')[0],i,accuracy)))   
            vxm.py.utils.save_volfile(p_zone.squeeze(), os.path.join(save_file, '%s_dice%d_%.4f_pz.nii.gz' % (name[0].split('.')[0],i, accuracy)))
            vxm.py.utils.save_volfile(t_zone.squeeze(), os.path.join(save_file, '%s_dice%d_%.4f_tz.nii.gz' % (name[0].split('.')[0],i, accuracy)))
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
        min1=np.median(np.where(res1[k,:]==np.amin(res1[k,:]))).astype(np.int)
        min2=np.median(np.where(res2[k,:]==np.amin(res2[k,:]))).astype(np.int)
        find_prec.append(prec[k*9+min1])
        find_reca.append(recall[k*9+min2])
    print('lesion: ', find_reca[0],find_prec[0],find_reca[1], find_prec[1], find_reca[2], find_prec[2])
    #print('lesion: ', np.sum(np.array(lesion_tp_num), axis=0)/np.sum(np.array(lesion_all),axis=0).tolist())
    #print('std: ',np.round(np.nanstd((np.array(lesion_tp_num)/np.array(lesion_all)), axis=0),4).tolist())
    #print(sum_accu[0] / len(accuracy_all), sum_accu[1] / (len(accuracy_all) - number_t), sum_accu[2] / (len(accuracy_all) - number_p), sum_accu[3]/len(accuracy_all), sum_accu[4]/(len(accuracy_all)),sum_accu[5]/(len(accuracy_all)))
