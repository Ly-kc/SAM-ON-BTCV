import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tqdm
import os
import open3d as o3d

id_to_label = {
    0: "background",
    1: "spleen",
    2: "rkid",
    3: "lkid",
    4: "gall",
    5: "eso",
    6: "liver",
    7: "sto",
    8: "aorta",
    9: "IVC",
    10: "veins",
    11: "pancreas",
    12: "rad",
    13: "lad"
}

id_to_color = np.loadtxt('../id_to_color.txt')

def vis_mask(mask_path):
    '''
    make mask image colorful
    '''
    mask = np.array(cv2.imread(mask_path,-1))  #(H,W)
    vis_mask = np.zeros(list(mask.shape) + [3])  #(H,W,3)
    print(np.unique(mask))
    for id in np.unique(mask):
        print((mask == id).shape)
        vis_mask[mask==id,:] = np.random.randint(0, 255, size=3)
    print(vis_mask.shape)    
    cv2.imwrite('../vis_mask.png', vis_mask)

def overlap_masks(img, masks):
    '''
    save overlaped masks into one image for visualizaion
    input:
        img:input image (H,W,3)
        masks:dict of sam output format----keys:(segmentation, area, predicted_iou)
    output: 
        colorful overlap masks in one image  (H,W,3)
    '''
    masks.sort(key=lambda x: x['area'], reverse=True)
    vis_img = np.zeros_like(img)
    for mask in masks:
        vis_img[mask['segmentation']] = np.random.randint(0, 255, size=3)
    
    return vis_img

def vis_semantic_masks(img, masks):
    '''
    save overlaped semantic masks into one image for visualizaion
    input:
        img:input image (H,W,3)
        masks:dict of sam output format----keys:(segmentation, area, predicted_iou,'pred_label_logits')
    output: 
        colorful overlap masks in one image  (H,W,3)
    '''
    # masks.sort(key=lambda x: x['area'], reverse=True)
    masks.sort(key=lambda x: x['predicted_iou'])
    vis_img = np.zeros_like(img)
    semantic_img = np.zeros([img.shape[0], img.shape[1]])
    for mask in masks:
        max_logits = max(mask['pred_label_logits'])
        label = mask['pred_label_logits'].index(max_logits)
        semantic_img[mask['segmentation']] = label
        vis_img[mask['segmentation']] = id_to_color[label]
    
    return semantic_img, vis_img

def plot_history_figure(log_dir, semantic=False):
    if(not semantic):
        loss_his_path = os.path.join(log_dir, 'loss_history.txt')
        dice_his_path = os.path.join(log_dir, 'dice_history.txt')
        plot_loss_figure(loss_his_path)
        plot_dice_figure(dice_his_path)
    else:
        plot_loss_figure(os.path.join(log_dir, 'dice_loss_history.txt'))
        plot_loss_figure(os.path.join(log_dir, 'semantic_loss_history.txt'))
        plot_dice_figure(os.path.join(log_dir, 'val_dice_history.txt'))
        plot_dice_figure(os.path.join(log_dir, 'val_acc_history.txt'))    
    

def plot_loss_figure(his_path = '../ckpts/loss_history.txt'):
    loss = np.loadtxt(his_path)
    plt.plot(loss)
    #添加坐标轴标签
    plt.xlabel('epoch')
    plt.ylabel('CrossEntropy loss')
    plt.savefig(his_path.replace('history.txt', 'history.png'))
    #清除图像，以防止叠加
    plt.clf()

def plot_dice_figure(his_path = '../ckpts/dice_history.txt'):
    dice = np.loadtxt(his_path)
    initial_dice = np.loadtxt('../ckpts/vit_h_2024-01-15-18-13-27/val_acc_history.txt')
    organ_count = np.array([408., 151., 168. ,168. , 51., 151., 285. ,181. ,381., 330. , 81. ,126. , 60. , 72.])
    dice = (dice/(organ_count+1e-8))
    dice = np.concatenate([initial_dice.reshape(1,-1), dice], axis=0)
    #共画13条线，每个器官一条
    #表格宽一点
    plt.figure(figsize=(8, 5))
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    
    for i in range(13):
        plt.plot(dice[:,i], label=id_to_label[i+1])
        #添加图例，位于表格外
        plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left', borderaxespad=0.) 
    plt.savefig(his_path.replace('history.txt', 'history.png'))  
    #清除图像，以防止叠加
    plt.clf()
    

def save_to_ply(result_dir_name, img_name, split='val'):
    '''
    将所有切片的semantic segmentation结果Lift到3D，使其成为点云，保存为ply文件
    result_dir_name: 保存semantic segmentation结果的文件夹
    img_name: 图像名字，如img0035
    split: val or test
    '''
    result_dir = os.path.join('../results', split, result_dir_name)

    mask_list = os.listdir(result_dir)
    mask_names = []
    for mask_name in mask_list:
        if(not 'maskraw' in mask_name):
            continue
        if(not img_name in mask_name):
            continue
        mask_names.append(mask_name)
        
    if(split in ['val', 'train']):
        nii_dir = '../data/Training/img'
    else:
        nii_dir = '../data/Testing/img'
    nii_path = os.path.join(nii_dir, img_name + '.nii.gz') 
    nii = sitk.ReadImage(nii_path)
    spacing = nii.GetSpacing()
    spacing = (1.0, spacing[1]/spacing[0],spacing[2]/spacing[0])

    #将mask转为三维点云
    pcd = o3d.geometry.PointCloud()
    points,colors = [],[]
    
    for mask_name in tqdm.tqdm(mask_names):
        mask_num = int(mask_name.split('_')[-1].split('.')[0])
        mask_path = os.path.join(result_dir, mask_name)
        mask = np.array(cv2.imread(mask_path, -1))
        pixelxs,pixelys = mask.nonzero()
        xs,ys = pixelxs*spacing[0],pixelys*spacing[1]
        zs = np.ones_like(xs) * (mask_num+70)*spacing[2]
        labels = mask[pixelxs,pixelys]
        mask_points = np.stack([xs,ys,zs], axis=1) #shape:(num_points, 3)
        mask_colors = id_to_color[labels]/255  #shape:(num_points, 3)
        points.append(mask_points)
        colors.append(mask_colors)
        
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    save_path = os.path.join(result_dir, img_name + '.ply')
    o3d.io.write_point_cloud(save_path, pcd)
    print('save to', save_path)
           

if __name__ == '__main__':
    # vis_mask('../data/processed/Training/All/mask0026_70.png')
    # save_to_ply(result_dir_name='semantic_30_vit_h_mask_decoder_200', img_name='img0035', split='val')
    # save_to_ply(result_dir_name='semantic_30_vit_h_mask_decoder_200_0.88', img_name='img0061', split='test')
    # plot_history_figure(log_dir = '../ckpts/vit_h_mask_2024-01-15-05-42-30')
    plot_history_figure(log_dir = '../ckpts/vit_h_2024-01-15-10-51-04', semantic=True)