import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt

import tqdm
from torch.optim import Adam

from preprocess_dataset import id_to_label
from visualize import *

from segment_anything import sem_sam_model_registry, safe_load_weights

from my_pridictor import MyPredictor
from btcv_dataset import BtcvDataset, btcv_collate_fn
from criterion import dice_loss


device = "cuda" if torch.cuda.is_available() else "cpu"

loss_history = []
dice_history = []

@torch.no_grad()
def validate(val_dataloader, my_predictor:MyPredictor, ccount_background=False):
    '''
    在验证集上计算每个器官的dice
    '''
    my_predictor.model.mask_decoder.eval()
    dice = np.zeros(14)
    organ_count = np.zeros(14)
    for img, embedding, label, prompts,batch_prompts, appearance in tqdm.tqdm(val_dataloader):
        label = label[0].to(device, non_blocking=True)
        img = img[0]
        prompts = prompts[0]
        appearance = appearance[0]
        embedding = embedding[0]
        #将13个器官的prompts组合成一个batch的prompts
        batch_prompts = batch_prompts[0]
        batch_prompts['multimask_output'] = False
        batch_prompts['return_logits'] = True
        my_predictor.set_image(img, image_embedding=embedding)
        masks, iou_predictions, low_res_masks, label_pred_logits = my_predictor.my_predict(**batch_prompts)
        for organ_id in range(14):
            if(not appearance[organ_id]):
                continue
            organ_mask = masks[organ_id]
            organ_label = label == organ_id
            organ_dice = 1 - dice_loss(organ_mask, organ_label)
            dice[organ_id] += organ_dice.item()
            organ_count[organ_id] += 1
            
    #log dice of all organ respectively and mean dice
    per_organ_dice = dice/(organ_count+1e-6)
    for organ_id in range(14):
        print(f'dice of {id_to_label[organ_id]}: ', per_organ_dice[organ_id])
    if(ccount_background):
        print('mean dice: ', per_organ_dice.mean())
    else:
        print('mean dice: ', per_organ_dice[1:].mean())
    dice_history.append(per_organ_dice)



def train_one_epoch(train_dataloader, my_predictor:MyPredictor, optimizer, data_weights=None):
    '''
    训练一个epoch(遍历所有图片一次)
    data_weights: (14,)  训练中这类器官的权重,一种简化的AdaBoost思想
    '''
    my_predictor.model.mask_decoder.train()
    if(data_weights is not None):
        print('data weights: ', data_weights)
    loss = None
    count = 0
    total_loss = 0
    img_count = 0
    for img, embedding, label, prompts,batch_prompts, appearance in tqdm.tqdm(train_dataloader):
        label = label[0].to(device, non_blocking=True)
        img = img[0]
        prompts = prompts[0]
        appearance = appearance[0]
        embedding = embedding[0]
        #将13个器官的prompts组合成一个batch的prompts
        batch_prompts = batch_prompts[0]
        batch_prompts['multimask_output'] = False
        batch_prompts['return_logits'] = True
        my_predictor.set_image(img, image_embedding=embedding)
        masks, iou_predictions, low_res_masks, label_pred_logits = my_predictor.my_predict(**batch_prompts)
        
        #对每个器官分别计算loss
        for organ_id in range(14):
            if(not appearance[organ_id]):
                continue
            organ_mask = masks[organ_id]
            organ_label = label == organ_id
            organ_loss = dice_loss(organ_mask, organ_label)
            # print(torch.unique(organ_mask), torch.unique(label), torch.unique(organ_label), organ_id, appearance[organ_id])
            if(data_weights is not None):
                organ_loss = organ_loss * data_weights[organ_id]
            if(loss is None):
                loss = organ_loss
            else:
                loss = loss + organ_loss
            count += 1
        
        #每5张图片更新一次参数
        if((img_count+1)%5 == 0 or (img_count+1)==len(train_dataloader)):
            loss = loss/count
            if((img_count+1)%100==0):
                print('training loss:', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count = 0
            loss = None
        img_count += 1
    loss_history.append(total_loss/(img_count+1e-6))
     
     
     
def calcu_data_weights(dices, need_backgrond=False):
    '''
    一种简化的AdaBoost思想
    param: dices: (14,)   各个器官validate的dice
    return: weights: (14,)  训练中这类器官的权重
    '''
    dis = np.array(1-dices)*5
    weights = np.exp(dis)
    if(not need_backgrond):
        weights[0] = 0
    weights = weights/weights.sum() * dices.shape[0]
    return weights
        
        
        
def train_all(from_pretrained = None, model_type='vit_b', checkpoint="../ckpts/sam_vit_b_01ec64.pth", use_data_weights=False):
    '''
    finetune一个sam模型的mask_decoder
    
    params: from_pretrained: (str) 已经训练好的mask_decoder的权重文件路径，需要与model_type对应
    params: model_type: (str) sam模型的类型
    params: checkpoint: (str) sam模型backbone的权重文件路径
    params:ues_data_weights: (bool) 是否使用data_weights
    '''
    sam = sem_sam_model_registry[model_type](checkpoint=checkpoint, need_semantic=False).cuda()
    if(from_pretrained is not None):
        print(f'load weights from {from_pretrained}')
        sam.mask_decoder = safe_load_weights(sam.mask_decoder, from_pretrained)
    my_predictor = MyPredictor(sam)
    
    optimizer = Adam(my_predictor.model.mask_decoder.parameters(), lr=5e-5)
    
    train_dataset = BtcvDataset(base_dir='../data/processed', split='train', prompt_class=['point'], model_type=model_type, need_backgrond=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=btcv_collate_fn,
        prefetch_factor=2,
    )
    val_dataset = BtcvDataset(base_dir='../data/processed', split='val', prompt_class=['point'], model_type=model_type, need_backgrond=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=btcv_collate_fn,
        prefetch_factor=2,
    )
    
    save_time = time.time()
    #将时间戳转换为时间
    timeArray = time.localtime(save_time)
    save_time = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    save_dir = f'../ckpts/{model_type}_mask_{save_time}'
    os.makedirs(save_dir, exist_ok=True)
    
    epoch_num = 200
    data_weights = None
    for epoch in range(epoch_num):
        print('epoch:', epoch+1)
        train_one_epoch(train_dataloader, my_predictor, optimizer, data_weights=data_weights)
        validate(val_dataloader,my_predictor)
        if(use_data_weights): 
            data_weights = calcu_data_weights(dice_history[-1])
        if((epoch+1)%20 == 0):
            torch.save(my_predictor.model.mask_decoder.state_dict(), os.path.join(save_dir, f'{model_type}_mask_decoder_{epoch+1}.pth'))
            torch.save(optimizer.state_dict(), '../ckpts/optimizer_{}.pth'.format(epoch))
    
    #save loss history
    np.savetxt(os.path.join(save_dir, 'loss_history.txt'), np.array(loss_history))
    np.savetxt(os.path.join(save_dir, 'dice_history.txt'), np.array(dice_history))    
    
    
    
if __name__ == '__main__':
    train_all(model_type='vit_h', checkpoint="../ckpts/sam_vit_h_4b8939.pth")
