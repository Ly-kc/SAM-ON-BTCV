import os
import numpy as np
import torch
import cv2
import tqdm


from myAutomaticMaskGenerator import MyAutomaticMaskGenerator
from segment_anything import sem_sam_model_registry, safe_load_weights

from visualize import *
 

def automask_all(split = 'val', 
                 model_type = 'vit_b', 
                 checkpoint = '../ckpts/sam_vit_b_01ec64.pth', 
                 decoder_checkpoint = '../ckpts/vit_b_2024-01-15-03-41-38/semantic_30_mask_decoder_300.pth',
                 nii_number=None):
    '''
    使用修改后能够预测器官类别的sam模型，对数据集某一split中的所有图片进行预测，每个切片的预测结果被保存为为像素值为0-13的semantic segmentation
    '''
    if(split in ['val', 'train']):
        data_dir = '../data/processed/Training/All'
    elif(split == 'test'):
        data_dir = '../data/processed/Testing/All'
    
    img_list = os.listdir(data_dir)
    img_names = []
    for img_name in img_list:
        if('img' not in img_name):
            continue
        img_num = int(img_name[3:7])
        if((split == 'train' and img_num >= 35) or split == 'val' and img_num < 35):
            continue
        if(nii_number is not None):
            if(img_num != nii_number):
                continue
        img_names.append(img_name)
        
    sam = sem_sam_model_registry[model_type](checkpoint=checkpoint).cuda()
    sam.mask_decoder = safe_load_weights(sam.mask_decoder, decoder_checkpoint)
    mask_generator = MyAutomaticMaskGenerator(sam,
                                              pred_iou_thresh=0.88,
                                              )
    
    # save_dir = os.path.join('../results', split, '000.png')
    save_dir = os.path.join('../results', split, decoder_checkpoint.split('/')[-1].split('.')[0] + '_0.88')
    os.makedirs(save_dir, exist_ok=True)
    for img_name in tqdm.tqdm(img_names):
        img_path = os.path.join(data_dir, img_name)
        automask(mask_generator, img_path, save_dir)
    
        
def automask(mask_generator:MyAutomaticMaskGenerator, img_path, save_dir):
    '''
    用grid points预测一张图片的masks, 并在vis_semantic_masks中将所有mask重叠到一张图片上，以获取semantic segmentationed结果
    '''
    img = np.array(cv2.imread(img_path))     

    masks = mask_generator.generate(img)
    semantic_img, vis_img = vis_semantic_masks(img, masks)

    save_semantic_path = os.path.join(save_dir, 'maskraw_' + os.path.basename(img_path))
    cv2.imwrite(save_semantic_path, semantic_img)

    save_vis_path = os.path.join(save_dir, 'colormask_' + os.path.basename(img_path))
    cv2.imwrite(save_vis_path, vis_img)
    
    
def calcu_mdice(result_dir_name='semantic_30_vit_h_mask_decoder_200', split='val'):
    '''
    用上述计算得到的semantic segmentation结果与gt计算dice
    params: result_dir_name: 保存semantic segmentation结果的文件夹
    '''
    result_dir = os.path.join('../results', split, result_dir_name)
    if(split == 'test'):
        label_dir = os.path.join('../data/processed', 'Testing', 'All')
    else:
        label_dir = os.path.join('../data/processed', 'Training', 'All')
    img_list = os.listdir(label_dir)
    img_names = []
    for img_name in img_list:
        if(not 'img' in img_name):
            continue
        img_num = int(img_name[3:7])
        if(split == 'train' and img_num >= 35 or split == 'val' and img_num < 35):
            continue
        img_names.append(img_name)        
    
    dices = np.zeros(14)
    counts = np.zeros(14)
    for img_name in tqdm.tqdm(img_names):
        gt_path = os.path.join(label_dir, img_name.replace('img', 'mask'))
        gt_label = np.array(cv2.imread(gt_path, -1))
        mask_name = 'maskraw_' + img_name
        mask_path = os.path.join(result_dir, mask_name)
        pred_label = np.array(cv2.imread(mask_path, -1))
        
        for label_id in range(14):
            gt_mask = gt_label == label_id
            if(gt_mask.sum() == 0):
                continue
            pred_mask = pred_label == label_id
            dice = 2 * (pred_mask * gt_mask).sum() / (pred_mask.sum() + gt_mask.sum())
            counts[label_id] += 1
            dices[label_id] += dice
    
    dices = (dices/(counts+1e-8))
    for label_id in range(14):
        print(f'{id_to_label[label_id]}: {dices[label_id]}')
    print('mean:', dices.mean())        

if __name__ == '__main__':
    # automask_all(split='val',model_type='vit_h', checkpoint="../ckpts/sam_vit_h_4b8939.pth", 
                #  decoder_checkpoint='../ckpts/vit_h_2024-01-15-10-51-04/semantic_30_vit_h_mask_decoder_200.pth')
    automask_all(split='test',model_type='vit_h', checkpoint="../ckpts/sam_vit_h_4b8939.pth", 
                 decoder_checkpoint='../ckpts/vit_h_2024-01-15-10-51-04/semantic_30_vit_h_mask_decoder_200.pth')
    # calcu_mdice()
    # automask_all(split='test',nii_number=61)