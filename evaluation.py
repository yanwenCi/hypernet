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

def main(args):
    path=args.sou_path
    dice_mean = []
    number_of_od = []
    number_of_pg = []
    number_of_pred = []
    gt_acc_lesion,od_tp,pred_acc, od_fp, gt_tp, pd_tp=0,0,0,0,0,0

    #path='/data0/yw/jupyter_folder/Attention-Gated-Network/experiment_unet_3mod80'

    t2w_list = glob.glob(path + '/*t2w.nii.gz')
    adc_list = [i.replace('t2w.nii.gz', 'adc.nii.gz') for i in t2w_list]
    dwi_list = [i.replace('t2w.nii.gz', 'dwi.nii.gz') for i in t2w_list]
    lab_list = [i.replace('t2w.nii.gz', 'label.nii.gz') for i in t2w_list]
    pre_list = [i.replace('_t2w.nii.gz', '.nii.gz') for i in t2w_list]
    pz_list =  [i.replace('t2w.nii.gz', 'pz.nii.gz') for i in t2w_list]
    tz_list =  [i.replace('t2w.nii.gz', 'tz.nii.gz') for i in t2w_list]
    for path_t2w, path_adc, path_dwi, path_lab, path_pre,path_tz,path_pz in zip(t2w_list, adc_list, dwi_list, lab_list, pre_list, tz_list,pz_list):
        if os.path.exists(path_adc) and os.path.exists(path_dwi) and os.path.exists(path_lab) and os.path.exists( path_pre):
            
            #print('Processing {}...'.format(os.path.split(path_pre)[-1]))
            # t2w = nib.load(path_t2w).get_fdata()
            # adc = nib.load(path_adc).get_fdata()
            # dwi = nib.load(path_dwi).get_fdata()
          
            lab = nib.load(path_lab).get_fdata()
            pre = nib.load(path_pre).get_fdata()
            p_zone=nib.load(path_pz).get_fdata()
            t_zone = nib.load(path_tz).get_fdata()

            hausd_dist, dice_vals,number_tplesion_gt, number_tplesion_pd, gt_number, pred_number=metrics(pre, lab, t_zone, p_zone)
            dice_mean.append([path_pre.split('/')[-1]]+hausd_dist.tolist() + dice_vals.tolist())
            #hausd_dist, dice_vals, precision, recal,number_tplesion_gt, number_tplesion_pd, od_acc_tp, od_acc_fp, gt_number, pred_number=metrics(pre, lab, t_zone, p_zone)
            #dice_mean.append([path_pre.split('/')[-1]]+hausd_dist.tolist() + dice_vals.tolist() + precision.tolist() + recal.tolist())
            number_of_pg.append([path_pre.split('/')[-1]]+number_tplesion_gt.tolist() + number_tplesion_pd.tolist())
            #number_of_od.append([path_pre.split('/')[-1]]+od_acc_tp.tolist()+ od_acc_fp.tolist())
            number_of_pred.append([path_pre.split('/')[-1]]+pred_number.tolist()+[gt_number])
            #od_tp+=od_acc_tp
            #od_fp+=od_acc_fp
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

    #with open(join(save_path, 'object.csv'), 'w+') as f_csv:
    #    write = csv.writer(f_csv)
    #    write.writerows(number_of_od)
    with open(join(save_path, 'PR.csv'), 'w+') as f_csv:
        write = csv.writer(f_csv)
        write.writerows(number_of_pg)
    with open(join(save_path,'pred_num.csv'), 'w+') as f_csv:
        write=csv.writer(f_csv)
        write.writerows(number_of_pred)

    f_csv.close()
    print(np.nanmean(np.array(dice_mean)[:,1:].astype(np.float),axis=0))
    #print('od', od_tp / (od_tp + od_fp), od_tp / gt_acc_lesion)
    print(gt_acc_lesion, pred_acc)
    print('gtpd', gt_tp/gt_acc_lesion, pd_tp/pred_acc)
    recall, prec=gt_tp/gt_acc_lesion, pd_tp/pred_acc
    crit1=[0.6]*9+[0.3]*9+[0.6]*9
    crit2=[0.4]*9+[0.15]*9+[0.4]*9
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

    print('find reca and prec: ', find_reca,find_prec)
        

        
        

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
def lesion_metric(pred, target_, t_zone, p_zone):
    iters=9*3
    od_gt_number=0
    pred_number=np.zeros(iters)
    number_tp_pd=np.zeros(iters)
    number_tp_gt=np.zeros(iters)
    pred_lesion=np.zeros(iters)
    gt_lesion=np.zeros(iters)
    for p in range(1,28):
        pred_seg=copy.deepcopy(pred)
        target=copy.deepcopy(target_)
        if p>9 and p<19:
            thre=(p-9)/10
            pred_seg=t_zone*pred_seg
            target=t_zone*target
        elif p>18:
            thre=(p-18)/10
            pred_seg=p_zone*pred_seg
            target=p_zone*target
        else:
            thre=p/10
        pred_seg[pred_seg>thre]=1
        pred_seg[pred_seg<=thre]=0
        thre=[0.05]
        overlap_pd, number_tp_pd[p-1], pred_lesion[p-1]=pn_rate(target,  pred_seg, thre,direct='pred')
        overlap_gt, number_tp_gt[p-1], gt_lesion[p-1]=pn_rate(target,  pred_seg,thre, direct='gt')
    return number_tp_gt, number_tp_pd, gt_lesion, pred_lesion


def metrics(pred, target_,t_zone,p_zone):
    iters=9*3
    od_acc_tp=np.zeros(iters)
    od_acc_fp=np.zeros(iters)
    od_gt_number=0
    pred_number=np.zeros(iters)
    hausd_dist=np.zeros(iters)
    dice_vals=np.zeros(iters)
    precision=np.zeros(iters)
    recal=np.zeros(iters)
    number_tp_pd=np.zeros(iters)
    number_tp_gt=np.zeros(iters)
    pred_lesion=np.zeros(iters)
    gt_lesion=np.zeros(iters)
    dice_score=vxm.losses.Dice(with_logits=False)
    for p in range(1,28):
        pred_seg=copy.deepcopy(pred)
        target=copy.deepcopy(target_)
        if p>9 and p<19:
            thre=(p-9)/10
            pred_seg=t_zone*pred_seg
            target=t_zone*target
        elif p>18:
            thre=(p-18)/10
            pred_seg=p_zone*pred_seg
            target=p_zone*target
        else:
            thre=p/10
        pred_seg[pred_seg>thre]=1
        pred_seg[pred_seg<=thre]=0

        dice_vals[p-1]=dice_score.loss(pred_seg.reshape((1,)+pred_seg.shape+(1,)), target.reshape((1,)+pred_seg.shape+(1,)))
        surf_dist=sd.compute_surface_distances(np.array(pred_seg, dtype=bool), np.array(target, dtype=bool), (1,1,1))
        hausd_dist[p-1]=sd.compute_robust_hausdorff(surf_dist,95)
        hausd_dist[np.isinf(hausd_dist)]=40
        #prec, rec = precision_and_recall(target, pred_seg,2)
        #precision[p-1]=prec[-1]
        #recal[p-1]=rec[-1]

        #calculate object detection metric
        #od_iou, number_lesion_tgt, number_lesion_pred, tp_num, fp_num=OD_PR(target, pred_seg, 0.25)
        #od_acc_tp[p-1]=tp_num
        #od_acc_fp[p-1]=fp_num
        thre=[0.1]
        overlap_pd, number_tp_pd[p-1], pred_lesion[p-1]=pn_rate(target,  pred_seg, thre,direct='pred')
        overlap_gt, number_tp_gt[p-1], gt_lesion[p-1]=pn_rate(target,  pred_seg,thre, direct='gt')

    return hausd_dist, dice_vals,number_tp_gt, number_tp_pd, gt_lesion, pred_lesion
    #return hausd_dist, dice_vals, precision, recal,number_tp_gt, number_tp_pd, od_acc_tp, od_acc_fp, gt_lesion, pred_lesion


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--sou_path", '-sp',default='/data0/yw/jupyter_folder/Attention-Gated-Network/experiment_unet_3mod80')
    parser.add_argument("--zone", choices=['pz','tz','prostate'])
    parser.add_argument('--gpu', type=str, default='0')
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    main(args)
