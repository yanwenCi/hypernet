import os
import glob
import nibabel as nib
import skimage.measure as measure
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main(path):
    #path=args.sou_path
    #path='/data0/yw/jupyter_folder/Attention-Gated-Network/experiment_unet_3mod80'
    lists = glob.glob(path + '/*.nii.gz')
    t2w_list = glob.glob(path + '/*t2w.nii.gz')
    adc_list = [i.replace('t2w.nii.gz', 'adc.nii.gz') for i in t2w_list]
    dwi_list = [i.replace('t2w.nii.gz', 'dwi.nii.gz') for i in t2w_list]
    lab_list = [i.replace('t2w.nii.gz', 'label.nii.gz') for i in t2w_list]
    pre_list = [i.replace('_t2w.nii.gz', '.nii.gz') for i in t2w_list]
    for path_t2w, path_adc, path_dwi, path_lab, path_pre in zip(t2w_list, adc_list, dwi_list, lab_list, pre_list):
        if os.path.exists(path_t2w):# and os.path.exists(path_dwi) and os.path.exists(path_lab) and os.path.exists(path_pre):
            print('Processing {}...'.format(os.path.split(path_pre)[-1]))
            t2w = nib.load(path_t2w).get_fdata()
            #adc = nib.load(path_adc).get_fdata()
            #dwi = nib.load(path_dwi).get_fdata()
            lab = nib.load(path_lab).get_fdata()
            pre = nib.load(path_pre).get_fdata()
            savepath = os.path.join(path,'./draw_contours/',path.split('/')[-1],os.path.split(path_pre)[-1].replace('.nii.gz', ''))
            
            print(savepath)
            for j in range(5,6):
                thres = j/10
                print(pre.shape,t2w.shape)
                drawcontour(pre.transpose(1, 0, 2), lab.transpose(1, 0, 2), t2w.transpose(1, 0, 2), thres,savepath, pref='t2w')
                #drawcontour(pre.transpose(1, 0, 2), lab.transpose(1, 0, 2), adc.transpose(1, 0, 2), thres,savepath, pref='adc')
                #drawcontour(pre.transpose(1, 0, 2), lab.transpose(1, 0, 2), dwi.transpose(1, 0, 2), thres,savepath, pref='dwi')
            # pre_image, number_of_pres=measure.label(pre, background = 0, return_num = True)
            # lab_image, number_of_labs = measure.label(lab, background=0, return_num=True)
            # for i, region in enumerate(measure.regionprops(pre_image, intensity_image=pre)):


def norm255(img):
    return 255*(img-img.min())/(img.max()-img.min())

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

def drawcontour(pre, lab, img,thres, save_path, pref):
    if not os.path.exists(save_path+'_'+str(thres)):
        os.makedirs(save_path+'_'+str(thres))
    pre=pre*1.0
    pre[pre>thres]=1
    pre[pre<=thres]=0
    x1, y1, z1 = np.nonzero(pre)
    x2, y2, z2 = np.nonzero(lab)
    z_slice=list(set(list(z1)+list(z2)))
    # must pre*1.0 
    pre=pre*1.0
    pre[pre>thres]=1
    pre[pre<=thres]=0


    #minz, maxz = [np.minimum(z1.min(), z2.min()), np.maximum(z1.max(), z2.max())]
    for i in z_slice[0::1]:
        # plt.imshow(t2w[:,:, i], cmap='gray')
        contours, hierarchy = cv2.findContours(np.array(pre[:, :, i] * 255, dtype=np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        contours2, hierarchy2 = cv2.findContours(np.array(lab[:, :, i] * 255, dtype=np.uint8), cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)

        contours2=removesamll(contours2)
        contours=removesamll(contours)

        image = norm255(img[:, :, i])
        image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image, contours, -1, color=(0, 255, 0), thickness=1)#pred green
        cv2.drawContours(image, contours2, -1, color=(0, 0, 255), thickness=1)#label red
        print('        contours is saved!')
        cv2.imwrite(os.path.join(save_path+'_'+str(thres), pref+'_slice'+str(i)+'.jpg'), image)  # Save the image



if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--sou_path", '-sp',default=None)
    args=parser.parse_args()
    dirs = os.listdir(args.sou_path)
    main(args.sou_path)
   # for _dir in dirs:
   #     path=os.path.join(args.sou_path, _dir, 'odds_hyp_cross1')
   #     print(path) 
   #     main(path)
