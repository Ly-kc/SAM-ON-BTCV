import SimpleITK as sitk
import numpy as np
import cv2
import os
from os.path import join
import tqdm

from visualize import id_to_label

base_dir = '../data'

def stat_center_width_for_organs(train_split_dir = '../data/Training'):
    """
    统计training set与validation set中每个器官窗位调整的center与width
    :return:
    centers:(14,)
    widths:(14,)  
    其中第0个数为所有前景窗口调整的参数,其余为各自器官的窗口调整参数
    """
    
    img_dir = join(train_split_dir, 'img')
    label_dir = join(train_split_dir, 'label')
    
    # 统计每个器官du的c平均enter与width
    center_sum = np.zeros(14)  #第零维为前景类
    width_sum = np.zeros(14)
    count = np.zeros(14)
    img_names = os.listdir(img_dir)
    for img_name in tqdm.tqdm(img_names):
        img_path = join(img_dir, img_name)
        label_path = join(label_dir, img_name.replace('img', 'label'))
        #读取nii
        imgs = sitk.ReadImage(img_path)
        imgs_numpy = sitk.GetArrayFromImage(imgs)
        labels = sitk.ReadImage(label_path)
        labels_numpy = sitk.GetArrayFromImage(labels)
        #统计
        for view in range(imgs_numpy.shape[0]):
            img_array = imgs_numpy[view]
            label_array = labels_numpy[view]
            for id in range(1,14):
                semantic_mask = label_array == id
                if(semantic_mask.sum() == 0):
                    continue
                max_hu,min_hu = np.max(img_array[semantic_mask]),np.min(img_array[semantic_mask])
                center_sum[id] += (max_hu + min_hu) / 2
                width_sum[id] += (max_hu - min_hu)
                count[id] += 1
                count[0] += 1
                center_sum[0] += (max_hu + min_hu) / 2
                width_sum[0] += (max_hu - min_hu)

    centers = np.divide(center_sum , (count+1e-6))
    widths = np.divide(width_sum , (count+1e-6))
    
    print('count:',count)
    print('centers:',centers)
    print('widths:',widths)
    
    np.savetxt(join(base_dir,'centers.txt'),centers)
    np.savetxt(join(base_dir,'widths.txt'),widths)
    return centers,widths
 
 
def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width and transform to [0,255]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    trunc_img = (ct_array - minWindow)/float(windowWidth)
    trunc_img[trunc_img < 0] = 0
    trunc_img[trunc_img > 1] = 1
    if not normal:
        trunc_img = (trunc_img *255).astype('uint8')
    return trunc_img


def process_to_gray(centers=None, widths=None, base_dir='../data',split='Training', discard=False):
    """
    将数据集中的图像根据窗口调整参数转换为灰度图
    :return:
    """
    if(centers is None):
        centers = np.loadtxt(join(base_dir,'centers.txt'))
    if(widths is None):
        widths = np.loadtxt(join(base_dir,'widths.txt'))
    save_base = join(base_dir, 'processed', split)
    
    split_dir = join(base_dir, split)
    img_dir = join(split_dir, 'img')
    label_dir = join(split_dir, 'label') if split != 'Testing' else None
    
    os.makedirs(save_base, exist_ok=True)
    
    img_names = os.listdir(img_dir)
    for img_name in tqdm.tqdm(img_names):
        img_path = join(img_dir, img_name)
        #读取nii
        imgs = sitk.ReadImage(img_path)
        imgs_numpy = sitk.GetArrayFromImage(imgs)
        if(split != 'Testing'):
            label_path = join(label_dir, img_name.replace('img', 'label'))
            labels = sitk.ReadImage(label_path)
            labels_numpy = sitk.GetArrayFromImage(labels)
        
        for view in range(imgs_numpy.shape[0]):
            img_array = imgs_numpy[view]
            save_dir = join(save_base,'All')
            if(split != 'Testing'):
                label_array = labels_numpy[view]
                if(discard and np.sum(label_array) == 0):
                    continue
                saved_label_path = join(save_dir, img_name.replace('img','mask').replace('.nii.gz', '_{}.png'.format(view)))
                cv2.imwrite(saved_label_path, label_array.astype('uint8'))
            
            #整张图像变换
            gray_img = window_transform(img_array, widths[0], centers[0])
            os.makedirs(save_dir, exist_ok=True)
            saved_img_path = join(save_dir, img_name.replace('.nii.gz', '_{}.png'.format(view)))
            # print(semantic_mask.astype('uint8'))
            cv2.imwrite(saved_img_path, gray_img)
            
            #各个器官分别变换
            for id in range(1,14):
                save_dir = join(save_base,id_to_label[id])
                if(split != 'Testing'):
                    semantic_mask = label_array == id
                    if(discard and semantic_mask.sum() == 0):
                        continue
                    saved_label_path = join(save_dir, img_name.replace('img','mask').replace('.nii.gz', '_{}.png'.format(view)))
                    cv2.imwrite(saved_label_path, semantic_mask.astype('uint8'))
                gray_img = window_transform(img_array, widths[id], centers[id])
                # trunc_img = trunc_img * semantic_mask
                #保存
                os.makedirs(save_dir, exist_ok=True)
                saved_img_path = join(save_dir, img_name.replace('.nii.gz', '_{}.png'.format(view)))
                # print(semantic_mask.astype('uint8'))
                cv2.imwrite(saved_img_path, gray_img)
        
            

if __name__ == '__main__':
    centers, widths = stat_center_width_for_organs()
    print(centers,widths)

    process_to_gray(base_dir='../data',split='Training', discard=True)
    process_to_gray(base_dir='../data',split='Testing')





