import numpy as np
import torch
import os
import time

from os.path import join
import tqdm
from torch.optim import Adam

from preprocess_dataset import id_to_label
from visualize import *

from segment_anything import sem_sam_model_registry, safe_load_weights

from my_pridictor import MyPredictor
from btcv_dataset import BtcvDataset, btcv_collate_fn
from criterion import dice_loss

CrossEntropyLoss = torch.nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"

#log信息
semantic_loss_history = []
dice_loss_history = []
val_dice_history = []
val_acc_history = []


@torch.no_grad()
def validate(val_dataloader, my_predictor:MyPredictor, ccount_background=False):
    '''
    在验证集上计算每个器官的dice以及acc
    '''
    my_predictor.model.mask_decoder.eval()
    dice = np.zeros(14)
    cross_entropy = np.zeros(14)
    acc = np.zeros(14)
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
            
            pred_label = torch.argmax(label_pred_logits[organ_id],dim=0)
            if(pred_label == organ_id):
                acc[organ_id] += 1
            # cross_entropy[organ_id] += CrossEntropyLoss(label_pred_logits[organ_id][None], torch.tensor([organ_id],device='cuda')).item()
            
            organ_count[organ_id] += 1
            
    print(organ_count)  
    #log dice of all organ respectively and mean dice
    per_organ_dice = dice/(organ_count+1e-6)
    per_organ_acc = acc/(organ_count+1e-6)
    for organ_id in range(14):
        print(f'dice of {id_to_label[organ_id]}: ', per_organ_dice[organ_id])
    if(ccount_background):
        print('mean dice: ', per_organ_dice.mean())
    else:
        print('mean dice: ', per_organ_dice[1:].mean())
    
    for organ_id in range(14):
        print(f'acc of {id_to_label[organ_id]}: ', per_organ_acc[organ_id])
    if(ccount_background):
        print('mean acc: ', per_organ_acc.mean())
    else:
        print('mean acc: ', per_organ_acc[1:].mean())
    val_dice_history.append(per_organ_dice)
    val_acc_history.append(per_organ_acc)



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
    total_mask_loss = 0
    total_semantic_loss = 0
    img_count = 0
    
    for img, embedding, label, prompts,batch_prompts, appearance in tqdm.tqdm(train_dataloader):
        label = label[0].to(device, non_blocking=True)
        img = img[0]
        prompts = prompts[0]
        appearance = appearance[0]
        embedding = embedding[0]
        #将13/14个器官的prompts组合成一个batch的prompts
        batch_prompts = batch_prompts[0]
        batch_prompts['multimask_output'] = False
        batch_prompts['return_logits'] = True
        my_predictor.set_image(img, image_embedding=embedding)
        masks, iou_predictions, low_res_masks, label_pred_logits = my_predictor.my_predict(**batch_prompts)
        #每个器官分别计算loss
        for organ_id in range(14):
            if(not appearance[organ_id]):
                continue
            organ_mask = masks[organ_id]
            organ_label = label == organ_id
            mask_loss = dice_loss(organ_mask, organ_label)   #mask loss
            semantic_loss = CrossEntropyLoss(label_pred_logits[organ_id][None], torch.tensor([organ_id],device='cuda'))  #semantic loss
            
            total_mask_loss += mask_loss.item()
            total_semantic_loss += semantic_loss.item()
            
            #混合loss
            organ_loss = mask_loss + 0.2*semantic_loss
            if(data_weights is not None):
                organ_loss = organ_loss * data_weights[organ_id]
            if(loss is None):
                loss = organ_loss
            else:
                loss = loss + organ_loss
            count += 1
            
        #当积攒了5个batch的loss时，进行一次反向传播
        if((img_count+1)%5 == 0 or (img_count+1)==len(train_dataloader)):
            loss = loss/count
            if((img_count+1)%100==0):
                print('training loss:', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = 0
            loss = None
        img_count += 1
    dice_loss_history.append(total_mask_loss/(img_count+1e-6))
    semantic_loss_history.append(total_semantic_loss/(img_count+1e-6))
     
     
     
def calcu_data_weights(dices, accs, need_backgrond=False):
    '''
    一种简化的AdaBoost思想
    param: dices: (14,)   各个器官validate的dice
    param: accs: (14,)   各个器官validate的acc
    return: weights: (14,)  训练中这类器官的权重
    '''
    dis = np.array(1-dices)*5 + np.array(1-accs)*5
    weights = np.exp(dis)
    if(not need_backgrond):
        weights[0] = 0
    weights = weights/weights.sum() * dices.shape[0]
    return weights
        
        
        
def train_all(from_pretrained = None, model_type='vit_b', checkpoint="../ckpts/sam_vit_b_01ec64.pth", use_data_weights=False):
    '''
    训练semantic mask decoder
    param: from_pretrained: 经过finetune后的mask decoder的权重的路径,需要与model_type对应
    param: model_type: sam的backbone的类型
    param: checkpoint: sam的backbone的权重的路径,需要与model_type对应
    param: use_data_weights: 是否使用一种类似于AdaBoost的思想,使得训练中的loss更加关注难以预测的器官
    '''
    sam = sem_sam_model_registry[model_type](checkpoint=checkpoint, need_semantic=True).cuda()
    if(from_pretrained is not None):
        print(f'load weights from {from_pretrained}')
        sam.mask_decoder = safe_load_weights(sam.mask_decoder, from_pretrained)
    my_predictor = MyPredictor(sam)
    
    optimizer = Adam(my_predictor.model.mask_decoder.parameters(), lr=5e-5)
    
    train_dataset = BtcvDataset(base_dir='../data/processed', split='train', prompt_class=['point'], model_type=model_type, need_backgrond=True, point_num=1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=btcv_collate_fn,
        prefetch_factor=2,
    )
    val_dataset = BtcvDataset(base_dir='../data/processed', split='val', prompt_class=['point'], model_type=model_type, need_backgrond=True, point_num=1)
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
    save_dir = f'../ckpts/{model_type}_{save_time}'
    os.makedirs(save_dir, exist_ok=True)

    epoch_num = 30
    data_weights = None
    
    for epoch in range(epoch_num):
        print('epoch:', epoch+1)
        train_one_epoch(train_dataloader, my_predictor, optimizer, data_weights=data_weights)
        validate(val_dataloader,my_predictor)
        if(use_data_weights):
            data_weights = calcu_data_weights(val_dice_history[-1], val_acc_history[-1], need_backgrond=True)
        if((epoch+1)%10 == 0):
            torch.save(my_predictor.model.mask_decoder.state_dict(), os.path.join(save_dir, f'semantic_{epoch+1}_{os.path.basename(from_pretrained)}'))
            torch.save(optimizer.state_dict(), '../ckpts/optimizer_{}.pth'.format(epoch))
            
    #save loss history
    np.savetxt(join(save_dir, 'semantic_loss_history.txt'), np.array(semantic_loss_history))
    np.savetxt(join(save_dir, 'dice_loss_history.txt'), np.array(dice_loss_history))
    np.savetxt(join(save_dir, 'val_dice_history.txt'), np.array(val_dice_history))
    np.savetxt(join(save_dir, 'val_acc_history.txt'), np.array(val_acc_history))
            
            
            
            
if __name__ == '__main__':
    # train_all('/home/lyz/ML-SAM-Project/ckpts/mask_decoder_300.pth')
    train_all(model_type='vit_h', checkpoint="../ckpts/sam_vit_h_4b8939.pth", 
              from_pretrained='../ckpts/vit_h_mask_2024-01-15-05-42-30/vit_h_mask_decoder_200.pth')
    
