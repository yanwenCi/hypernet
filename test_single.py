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
from metrics import pn_rate
import surface_distance as sd
# from tqdm.keras import TqdmCallback
# tf.compat.v1.disable_eager_execution()

from tensorflow.python.framework.ops import disable_eager_execution

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

# loss hyperparameters
parser.add_argument('--type', type=int, default=None)
parser.add_argument('--activ',  default='sigmoid')
parser.add_argument('--hyper_num', type=int, default=3)

parser.add_argument('--image-loss', default='dice',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--image-sigma', type=float, default=0.05,
                    help='estimated image noise for mse image scaling (default: 0.05)')
parser.add_argument('--oversample-rate', type=float, default=1,
                    help='hyperparameter end-point over-sample rate (default 0.2)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpu_avilable = tf.config.experimental.list_physical_devices('GPU')
print(gpu_avilable)


logdir = args.model_dir + "/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel
# scan-to-scan generator
base_generator = vxm.generators.single_mods_gen(
    args.img_list, phase='test', batch_size=args.batch_size, test= True, add_feat_axis=add_feat_axis, type=args.type)
# random hyperparameter generator


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
accuracy_all,lesion_all,lesion_tp_num=[],[],[]
# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
save_file = os.path.join(args.pred_dir, args.model_dir.split('/')[-1])
if not os.path.exists(save_file):
    os.makedirs(save_file)
results=[]
with tf.device(device):
    # build the model
    model = vxm.networks.UnetSingle(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        src_feats=nfeats,
        trg_feats=1,
        unet_half_res=False,
        activate=args.activ)

    print(model.summary())
    # load initial weights (if provided)

    model.load_weights(os.path.join(model_dir, '{:04d}.h5'.format(int(args.load_weights))))
    print('loading weights from {:04d}.h5'.format(int(args.load_weights)))

    # prepare image loss
    accuracy_func=vxm.losses.Dice(with_logits=False)
    number_p, number_t=0,0
    # prepare loss functions and compile model
    for i, data in enumerate(base_generator):
        inputs, outputs, zone, name = data
        predicted = model.predict(inputs)[-1]
        # predicted = (predicted-predicted.min())/(predicted.max()-predicted.min())
        # print(predicted.max())
        p_zone = np.zeros_like(outputs[0])
        p_zone[zone == 1] = 1
        t_zone = np.zeros_like(outputs[0])
        t_zone[zone == 2] = 1
        # import matplotlib.pyplot as plt
        # plt.imshow(p_zone[0,:,:,46,0])
        # plt.show()
        p_lesion = outputs[0] * p_zone
        t_lesion = outputs[0] * t_zone
        if np.sum(p_lesion) < 27:
            number_p += 1
            # print('    %s p zone has no lesion' % name[0])
        if np.sum(t_lesion) < 27:
            number_t += 1
            # print('    %s t zone has no lesion' % name[0])
        p_predict = predicted.round() * p_zone
        t_predict = predicted.round() * t_zone
        accuracy = accuracy_func.loss(outputs[0], predicted.round())
        accuracy_p = accuracy_func.loss(p_lesion, p_predict)
        accuracy_t = accuracy_func.loss(t_lesion, t_predict)
        
        surf_dist=sd.compute_surface_distances(np.array(predicted.squeeze().round(), dtype=bool), np.array(outputs[0].squeeze(), dtype=bool), (1,1,1))
        hausd_dist=sd.compute_robust_hausdorff(surf_dist,95)
        hausd_dist=min(50 ,hausd_dist)
        surf_dist_tz=sd.compute_surface_distances(np.array(t_predict.squeeze(), dtype=bool), np.array(t_lesion.squeeze(), dtype=bool), (1,1,1))
        hausd_dist_tz=sd.compute_robust_hausdorff(surf_dist_tz,95)
        hausd_dist_tz=min(50,hausd_dist_tz)
        surf_dist_pz=sd.compute_surface_distances(np.array(p_predict.squeeze(), dtype=bool), np.array(p_lesion.squeeze(), dtype=bool), (1,1,1))
        hausd_dist_pz=sd.compute_robust_hausdorff(surf_dist_pz,95)
        hausd_dist_pz=min(50,hausd_dist_pz)
        accuracy_all.append([accuracy, accuracy_t, accuracy_p,hausd_dist,hausd_dist_tz,hausd_dist_pz])
        
        # lesion level
        thresh=[0.25]
        overlap_pd_tz, number_tp_pd_tz, pred_lesion_tz=pn_rate(t_lesion.squeeze(),t_predict.squeeze(), thresh ,direct='pred')
        overlap_gt_tz, number_tp_gt_tz, gt_lesion_tz=pn_rate(t_lesion.squeeze(), t_predict.squeeze(),thresh, direct='gt')
        overlap_pd_pz, number_tp_pd_pz, pred_lesion_pz=pn_rate(p_lesion.squeeze(),p_predict.squeeze(), thresh ,direct='pred')
        overlap_gt_pz, number_tp_gt_pz, gt_lesion_pz=pn_rate(p_lesion.squeeze(), p_predict.squeeze(),thresh, direct='gt')
        overlap_pd, number_tp_pd, pred_lesion=pn_rate(outputs[0].squeeze(),predicted.round().squeeze(), thresh ,direct='pred')
        overlap_gt, number_tp_gt, gt_lesion=pn_rate(outputs[0].squeeze(), predicted.round().squeeze(), thresh, direct='gt')


        if np.sum(p_lesion)<27:
            accuracy_p=np.nan
            hasud_dist_pz=np.nan
        if np.sum(t_lesion)<27:
            accuracy_t=np.nan
            hausd_dist_tz=np.nan
        accuracy_all.append([accuracy, accuracy_t, accuracy_p,hausd_dist,hausd_dist_tz,hausd_dist_pz])
        lesion_all.append([gt_lesion, pred_lesion, gt_lesion_tz,pred_lesion_tz,gt_lesion_pz,pred_lesion_pz])
        lesion_tp_num.append([number_tp_gt,number_tp_pd, number_tp_gt_tz,number_tp_pd_tz,number_tp_gt_pz,number_tp_pd_pz])

        #print('  ',name[0], accuracy, accuracy_t, accuracy_p)

        if i % 1 == 0:
            seg_result = predicted.squeeze()
            # print('%d-th mean accuracy: %f' % (i, np.array(accuracy_all).mean(axis=0)))
            vxm.py.utils.save_volfile(seg_result,
                                      os.path.join(save_file, '%s_dice_%.4f_pred.nii.gz' % (name[0].split('.')[0], accuracy)))
            vxm.py.utils.save_volfile(inputs[0][:,0,...].squeeze(), os.path.join(save_file, '%s_dice_%.4f_t2w.nii.gz' % (name[0].split('.')[0], accuracy)))
            vxm.py.utils.save_volfile(inputs[0][:,2,...].squeeze(), os.path.join(save_file, '%s_dice_%.4f_dwi.nii.gz' % (name[0].split('.')[0], accuracy)))
            vxm.py.utils.save_volfile(inputs[0][:,1,...].squeeze(), os.path.join(save_file, '%s_dice_%.4f_adc.nii.gz' % (name[0].split('.')[0], accuracy)))
            vxm.py.utils.save_volfile(inputs[0].squeeze(), os.path.join(save_file, '%s_dice_%.4f.nii.gz' % (name[0].split('.')[0], accuracy)))
            vxm.py.utils.save_volfile(p_zone.squeeze(), os.path.join(save_file, '%s_dice_%.4f_pz.nii.gz' % (name[0].split('.')[0], accuracy)))
            vxm.py.utils.save_volfile(t_zone.squeeze(), os.path.join(save_file, '%s_dice_%.4f_tz.nii.gz' % (name[0].split('.')[0], accuracy)))
            vxm.py.utils.save_volfile(outputs[0].squeeze(),  os.path.join(save_file, '%s_dice_%.4f_label.nii.gz'% (name[0].split('.')[0],accuracy)))
    sum_accu = np.array(accuracy_all).sum(axis=0)
    
    print('std: ',np.round(np.nanstd(np.array(accuracy_all), axis=0),4))
    print('mean: ',np.round(np.nanmean(np.array(accuracy_all),axis=0),4))
    print('lesion: ', np.sum(np.array(lesion_tp_num), axis=0)/np.sum(np.array(lesion_all),axis=0    ))
    print('std: ',np.round(np.nanstd((np.array(lesion_tp_num)/np.array(lesion_all)), axis=0),4))
    #print(sum_accu[0] / len(accuracy_all), sum_accu[1] / (len(accuracy_all) - number_t),
#          sum_accu[2] / (len(accuracy_all) - number_p), sum_accu[3]/len(accuracy_all),sum_accu[4]/(len(accuracy_all)),sum_accu[5]/(len(accuracy_all)))

