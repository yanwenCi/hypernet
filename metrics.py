# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
#import cv2
import torch
from torch import nn
import torch.nn.functional as F
import skimage.measure as measure
import skimage.morphology as morphology

def OD_PR(tgt, img, thre):
    # ltgt is label, img is prediction
    iou_metric=[]
    tot_lesion_tgt, tp_number, fp_number=(0,0,0)
    label_tgt, number_of_tgts=measure.label(tgt, background = 0, return_num = True, connectivity=1)
    label_pred, number_of_preds=measure.label(img, background = 0, return_num = True, connectivity=1)
    tp_list=[]
    #print(tgt.max(),tgt.min(), img.max(),img.min())
    tot_lesion_pred=number_of_preds*1
    # eliminate  small lesion in prediction
    for i,region2 in zip(range(number_of_preds),measure.regionprops(label_pred, intensity_image=img)): 
        if region2.area<9:
            tot_lesion_pred=tot_lesion_pred-1
    # computing precision and recall
    for j,region in zip(range(number_of_tgts),measure.regionprops(label_tgt, intensity_image=tgt)):
        lesion_tgt=np.zeros(tgt.shape)
        if region.area<9:
            continue
            # eliminate small lesion in gt
        else:
            tot_lesion_tgt+=1

            for i,region2 in zip(range(number_of_preds),measure.regionprops(label_pred, intensity_image=img)):
                lesion_pred=np.zeros(img.shape)
                if region2.area<9:
                    
                    continue
                elif i in tp_list:
                    continue
                    # tp lesion is counted once only
                else:
                    lesion_tgt[label_tgt==j+1]=1
                    lesion_pred[label_pred==i+1]=1
                    #lesion_tgt[minx:maxx, miny:maxy, minz:maxz]=label_tgt[minx:maxx, miny:maxy, minz:maxz]
                    #lesion_pred[minx2:maxx2, miny2:maxy2, minz2:maxz2]=label_pred[minx2:maxx2, miny2:maxy2, minz2:maxz2]
                    iou=OD_IOU(lesion_tgt, lesion_pred)
                    #print(lesion_tgt.min(),lesion_tgt.max(), lesion_pred.max(),lesion_pred.min())
                    if iou is not None:
                        iou_metric.append(iou)
                        if iou>thre:
                            tp_number+=1
                            tp_list.append(i)
    fp_number=tot_lesion_pred-tp_number
    return np.array(iou_metric), tot_lesion_tgt,number_of_preds, tp_number, fp_number
                   

def lesion_size(tgt, img):
    # ltgt is label, img is prediction
    dice, size_gt, size_pred = [], [], []
    label_tgt, number_of_tgts=measure.label(tgt, background = 0, return_num = True, connectivity=1)
    label_pred, number_of_preds=measure.label(img, background = 0, return_num = True, connectivity=1)
    tp_list=[]
    # eliminate  small lesion in prediction
    # computing precision and recall
    for j,region in zip(range(number_of_tgts),measure.regionprops(label_tgt, intensity_image=tgt)):
        lesion_tgt=np.zeros(tgt.shape)
        if region.area<9:
            continue
            # eliminate small lesion in gt
        else:
            for i,region2 in zip(range(number_of_preds),measure.regionprops(label_pred, intensity_image=img)):
                lesion_pred=np.zeros(img.shape)
                if region2.area<9:

                    continue
                    # tp lesion is counted once only
                elif i in tp_list:
                    continue
                else:
                    lesion_tgt[label_tgt==j+1]=1
                    lesion_pred[label_pred==i+1]=1
                    dice.append((2*np.sum(lesion_tgt*lesion_pred))/(np.sum(lesion_tgt)+np.sum(lesion_pred)))
                    if dice[-1]>0.1:
                        tp_list.append(i)
                    size_gt.append(np.sum(lesion_tgt))
                    size_pred.append(np.sum(lesion_pred))
    return size_pred, size_gt, dice


def OD_IOU(tgt, pred):
    tgt=tgt.reshape(1,-1)
    pred=pred.reshape(1,-1)
    tp=np.sum(tgt*pred, -1)
    if tp>0:
        
        fn=np.sum(tgt*(1-pred), -1)
        fp=np.sum((1-tgt)*pred, -1)
      
        return tp/(tp+fn+fp)

def pn_rate(tgt, img, thre, direct='gt'):
    #img and tgt: 3d np array
    #tgt is label, img is predication
    tgt=tgt.astype(bool)
    img=img.astype(bool)
    tgt=morphology.remove_small_holes(morphology.remove_small_objects(tgt,26, connectivity=3),26,connectivity=3)
    img=morphology.remove_small_holes(morphology.remove_small_objects(img,26, connectivity=3),26,connectivity=3)

    if direct=='pred':
        label_image, number_of_labels=measure.label(img, background = 0, return_num = True, connectivity=1)
        y_last=tgt
    elif direct=='gt':
        y_last=img
        label_image, number_of_labels=measure.label(tgt, background = 0, return_num = True, connectivity=1)
    measures=np.empty(number_of_labels*2)
    i=0
    th_count=[0]*(len(thre))
    number_of_labels_=number_of_labels+0
    for j,region in zip(range(number_of_labels),measure.regionprops(label_image, intensity_image=img)):
        lesion=np.zeros(tgt.shape)
        if region.area<9:
            measures=measures[:-2]
            number_of_labels_=number_of_labels_-1
        else:
            minx, miny, minz, maxx, maxy, maxz = region.bbox
            #lesion[minx:maxx, miny:maxy, minz:maxz]=label_image[minx:maxx, miny:maxy, minz:maxz]
            lesion[label_image==j+1]=1
            #print(label_image.min(), label_image.max())
            measure1, measure2=calculate_pn(lesion, y_last)
            #measure1, measure2, measure3=calculate_pn(img[minx:maxx, miny:maxy, minz:maxz], tgt[minx:maxx, miny:maxy, minz:maxz])
            measures[2*i]=measure1
            measures[2*i+1]=measure2
            i+=1
            for k, th in enumerate(thre):
                if measure1>th:
                    th_count[k]+=1

    return measures, th_count[0], number_of_labels_



def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

class DiceMetric(nn.Module):
    def __init__(self, with_logic=True):
        super(DiceMetric, self).__init__()
        self.with_logic=with_logic
    def forward(self, warped, target):
        smooth = 0.01
        batch_size = warped.size(0)
        if self.with_logic:
            warped = torch.sigmoid(warped).view(batch_size, -1)
        else:
            warped = warped.view(batch_size, -1)
        target = target.view(batch_size, -1)

        inter = torch.sum(warped * target, 1) + smooth
        union = torch.sum(warped, 1) + torch.sum(target, 1) + smooth
        score = torch.sum(2.0 * inter / union)
        score = score / float(batch_size)
        return score


def CC3D(I, J):
    win = [9, 9, 9]
    I = I.float()
    J = J.float()
    I2 = I * I
    J2 = J * J
    IJ = I * J
    device = I.get_device()
    filt = torch.ones([1, 1, win[0], win[1], win[2]]).to(device)
    filt = torch.ones([1, 1, win[0], win[1], win[2]]).to(device)

    I_sum = F.conv3d(I, filt, stride=1, padding=4) / 5.0
    J_sum = F.conv3d(J, filt, stride=1, padding=4) / 5.0
    I2_sum = F.conv3d(I2, filt, stride=1, padding=4) / 5.0
    J2_sum = F.conv3d(J2, filt, stride=1, padding=4) / 5.0
    IJ_sum = F.conv3d(IJ, filt, stride=1, padding=4) / 5.0

    win_size = win[0] * win[1] * win[2]
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-20)
    cc_l = -1.0 * torch.mean(cc)
    return {'overall_acc': cc_l}

def calculate_pn(y_pred, y_true):
    # pred true ordered
    # lesion is considered as pred, then for that lesion, fp+tp=lesion
    y_pred=y_pred.reshape(1,-1)
    y_true = y_true.reshape(1, -1)        
    
    tp = np.sum(y_true * y_pred, axis=-1)
    lesion_sum=np.sum(y_pred, axis=-1)
    tn = np.sum((1 - y_true) * (1 - y_pred), axis=-1)
    fp = np.sum((1 - y_true) * y_pred, axis=-1)
    fn = np.sum(y_true * (1 - y_pred), axis=-1)
    return tp/(tp+fp), fp/(fp+tp)
    #if direct=='gt':
    #    return tp/(tp+fn), fn/(fn+tp)
    #elif direct=='pred':
    #    return tp/(tp+fp), fp/(tp+fp)

def gradientpenalty(penalty='l1'):
    def loss( y_pred):
        dy = torch.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx)+torch.mean(dy)+torch.mean(dz)
        return d/3.0
    return {'overall_acc': loss}

def segmentation_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {'overall_acc': acc,
            'mean_acc': acc_cls,
            'freq_w_acc': fwavacc,
            'mean_iou': mean_iu}



def dice_score_list(label_gt, label_pred, n_class):
    """

    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)


def dice_score(label_gt, label_pred, n_class):
    """

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """

    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
    for class_id in range(n_class):
        img_A = np.array(label_gt == class_id, dtype=np.float32).flatten()
        img_B = np.array(label_pred == class_id, dtype=np.float32).flatten()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        dice_scores[class_id] = score

    return dice_scores


def precision_and_recall(label_gt, label_pred, n_class):
    from sklearn.metrics import precision_score, recall_score
    assert len(label_gt) == len(label_pred)
    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()
    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))

    return precision, recall


def distance_metric(seg_A, seg_B, dx, k):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
        """

    # Extract the label k from the segmentation maps to generate binary maps
    seg_A = (seg_A == k)
    seg_B = (seg_B == k)

    table_md = []
    table_hd = []
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            _, contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            _, contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
            table_md += [md]
            table_hd += [hd]

    # Return the mean distance and Hausdorff distance across 2D slices
    mean_md = np.mean(table_md) if table_md else None
    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_md, mean_hd

if __name__ == '__main__':
    import sys
    sys.path.append('../dataio/loader')
    from utils import load_nifti_img
    data,meta=load_nifti_img('/data0/yw/registration-3d/data/iso_seg/case3.nii', dtype=np.uint8)
    pred, _ =load_nifti_img('/data0/yw/registration-3d/data/iso_seg/case13.nii', dtype=np.uint8)
    score = segmentation_scores(data, pred, 3)
