from os.path import join
import os
import glob
import nibabel as nib
import skimage.measure as measure
import numpy as np
import matplotlib.pyplot as plt
import cv2
from metrics import precision_and_recall, DiceMetric, dice_score, pn_rate, OD_PR
import surface_distance as sd
import csv
import copy
import voxelmorph as vxm
import os


def main(args):
    path=args.sou_path
    dice_mean = []
    number_of_od = []
    number_of_pg = []
    number_of_pred = []
    gt_acc_lesion,od_tp,pred_acc, od_fp, gt_tp, pd_tp=0,0,0,0,0,0

    #path='/data0/yw/jupyter_folder/Attention-Gated-Network/experiment_unet_3mod80'

    lab_list = glob.glob(path + '/*label.nii.gz')

    t2w_list = [i.replace('label.nii.gz', 'pred.nii.gz') for i in lab_list]
    pre_list = [i.replace('_t2w.nii.gz', '.nii.gz') for i in t2w_list]
    tz_list = [i.replace('label.nii.gz', 'tz.nii.gz') for i in lab_list]
    pz_list = [i.replace('label.nii.gz', 'pz.nii.gz') for i in lab_list]
    for path_t2w, path_lab, path_pre, path_tz, path_pz in zip(t2w_list, lab_list, pre_list, tz_list,pz_list):

            print('Processing {}...'.format(os.path.split(path_pre)[-1]))
            # t2w = nib.load(path_t2w).get_fdata()
            # adc = nib.load(path_adc).get_fdata()
            # dwi = nib.load(path_dwi).get_fdata()
            lab = nib.load(path_lab).get_fdata()
            pre = nib.load(path_pre).get_fdata()
            if args.zone=='pz':
                zone=nib.load(path_pz).get_fdata()
                lab=zone*lab
                pre=zone*pre
            elif args.zone=='tz':
                zone = nib.load(path_tz).get_fdata()    
                lab=zone*lab
                pre=zone*pre
            hausd_dist, dice_vals, precision, recal,number_tplesion_gt, number_tplesion_pd, od_acc_tp, od_acc_fp, gt_number, pred_number=metrics(pre, lab)
            dice_mean.append([path_pre.split('/')[-1]]+hausd_dist.tolist() + dice_vals.tolist() + precision.tolist() + recal.tolist())
            number_of_pg.append([path_pre.split('/')[-1]]+number_tplesion_gt.tolist() + number_tplesion_pd.tolist())
            number_of_od.append([path_pre.split('/')[-1]]+od_acc_tp.tolist()+ od_acc_fp.tolist())
            number_of_pred.append([path_pre.split('/')[-1]]+pred_number.tolist()+[gt_number])
            od_tp+=od_acc_tp
            od_fp+=od_acc_fp
            pred_acc+=pred_number
            gt_acc_lesion+=gt_number

            gt_tp+=number_tplesion_gt
            pd_tp+=number_tplesion_pd
    save_path=join(path, 'metrics')
    print(od_tp, od_fp, pred_acc)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(save_path)
    with open(join(save_path, 'test_loss_log.csv'), 'w+') as f_csv:
        write = csv.writer(f_csv)
        write.writerows(dice_mean)

    with open(join(save_path, 'object.csv'), 'w+') as f_csv:
        write = csv.writer(f_csv)
        write.writerows(number_of_od)
    with open(join(save_path, 'PR.csv'), 'w+') as f_csv:
        write = csv.writer(f_csv)
        write.writerows(number_of_pg)
    with open(join(save_path,'pred_num.csv'), 'w+') as f_csv:
        write=csv.writer(f_csv)
        write.writerows(number_of_pred)

    f_csv.close()
    print(np.mean(np.array(dice_mean)[:,1:].astype(np.float),axis=0))
    print('od', od_tp / (od_tp + od_fp), od_tp / gt_acc_lesion)
    print(gt_acc_lesion)
    print('gtpd', gt_tp/gt_acc_lesion, pd_tp/pred_acc)

def removesamll(contours, thres=0.5):
    n = len(contours)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= thres:
            cv_contours.append(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue
    print('    Founding {} contours'.format(len(cv_contours)))
    return cv_contours

def metrics(pred, target):
    od_acc_tp=np.zeros(9)
    od_acc_fp=np.zeros(9)
    od_gt_number=0
    pred_number=np.zeros(9)
    hausd_dist=np.zeros(9)
    dice_vals=np.zeros(9)
    precision=np.zeros(9)
    recal=np.zeros(9)
    number_tp_pd=np.zeros(9)
    number_tp_gt=np.zeros(9)
    pred_lesion=np.zeros(9)
    gt_lesion=np.zeros(9)
    dice_score=vxm.losses.Dice(with_logits=False)
    for p in range(1,10):
        pred_seg=copy.deepcopy(pred)
        thre=p/10
        pred_seg[pred_seg>thre]=1
        pred_seg[pred_seg<=thre]=0

        dice_vals[p-1]=dice_score.loss(pred_seg.reshape((1,)+pred_seg.shape+(1,)), target.reshape((1,)+pred_seg.shape+(1,)))
        surf_dist=sd.compute_surface_distances(np.array(pred_seg, dtype=bool), np.array(target, dtype=bool), (1,1,1))
        hausd_dist[p-1]=sd.compute_robust_hausdorff(surf_dist,95)
        hausd_dist[np.isinf(hausd_dist)]=50
        prec, rec = precision_and_recall(target, pred_seg,2)
        precision[p-1]=prec[-1]
        recal[p-1]=rec[-1]

        #calculate object detection metric
        od_iou, number_lesion_tgt, number_lesion__pred, tp_num, fp_num=OD_PR(target, pred_seg, 0.25)
        od_acc_tp[p-1]=tp_num
        od_acc_fp[p-1]=fp_num
        thre=[0.25]
        overlap_pd, number_tp_pd[p-1], pred_lesion[p-1]=pn_rate(target,  pred_seg, thre,direct='pred')
        overlap_gt, number_tp_gt[p-1], gt_lesion[p-1]=pn_rate(target,  pred_seg,thre, direct='gt')

    return hausd_dist, dice_vals, precision, recal,number_tp_gt, number_tp_pd, od_acc_tp, od_acc_fp, number_lesion_tgt, pred_lesion


if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--sou_path", '-sp',default='/data0/yw/jupyter_folder/Attention-Gated-Network/experiment_unet_3mod80')
    parser.add_argument("-gpu", type=str, default="0")
    parser.add_argument('-zone',choices=['pz','tz','pr'])
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
