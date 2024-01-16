import torch
from torch.utils.data import Dataset
import os
from os.path import join
import cv2
import numpy as np
import tqdm

from my_pridictor import MyPredictor
from segment_anything import SamPredictor, sam_model_registry

class BtcvDataset(Dataset): 
    def __init__(self, 
                 base_dir, 
                 split='train',
                 prompt_class=['point'],
                 model_type='vit_b', 
                 checkpoint=None,
                 need_backgrond=False,
                 point_num=2
                 ): 
        self.split = split
        if(split in ['train','val']):
            self.data_dir = join(base_dir, 'Training', 'All')
        else:
            self.data_dir = join(base_dir, 'Testing', 'All')
        img_names = os.listdir(self.data_dir)
        img_names = self.filter_img(img_names)
        self.img_name_list = img_names
        self.prompt_class = prompt_class
        self.embed_computed = False
        self.model_type = model_type
        if(checkpoint is not None):
            self.checkpoint = checkpoint
        elif(model_type == 'vit_b'):
            self.checkpoint = '../ckpts/sam_vit_b_01ec64.pth'
        elif(model_type == 'vit_h'):
            self.checkpoint = '../ckpts/sam_vit_h_4b8939.pth'
        
        self.need_backgrond = need_backgrond
        self.point_num = point_num
    
    def filter_img(self, img_names):
        filtered_img_names = []
        for img_name in img_names:
            if('img' not in img_name):
                continue
            img_num = int(img_name[3:7])
            if((self.split == 'train' and img_num >= 35) or self.split == 'val' and img_num < 35):
                continue
            filtered_img_names.append(img_name)
        return filtered_img_names
                
    def precompute_embeddings(self, embedding_dir = '../data/sam_embedding', model_type = 'vit_b', checkpoint = '../ckpts/sam_vit_b_01ec64.pth'):
        sam = sam_model_registry[model_type](checkpoint=checkpoint).cuda()
        predictor = SamPredictor(sam)
        save_dir = join(embedding_dir, model_type)
        os.makedirs(save_dir, exist_ok=True)
        for img_name in tqdm.tqdm(self.img_name_list, desc='computing embeddings'):
            embedding_path = join(save_dir, img_name.replace('.png','.pt'))
            if(os.path.exists(embedding_path)):
                continue    
            img_path = join(self.data_dir, img_name)
            img = np.array(cv2.imread(img_path))
            predictor.set_image(img)
            embedding = predictor.get_image_embedding()
            torch.save(embedding, embedding_path) 
        self.embed_computed = True    
    
    def get_embedding(self, img_name, embedding_dir = '../data/sam_embedding'):
        if(not self.embed_computed):
            self.precompute_embeddings(embedding_dir, self.model_type, self.checkpoint)
        return torch.load(join(embedding_dir, self.model_type, img_name.replace('.png','.pt')))
    
    def __len__(self):#返回整个数据集图片的数目
        return len(self.img_name_list)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        '''
        :param index: (int) 索引,代表将要取出list中第几张图片
        :return img: np.ndarray(H,W);取出的灰度图
        :return label: np.ndarray(H,W);取出的图片的label
        :return prompts: list[dict{},dict{}...].每个dict代表图片中一个器官的prompt,dict的格式参考criterion.py的第76行
        :return appearance: np.ndarray(14,dtype=bool). 图片中是否存在每个器官的gt, 第0维是背景
        '''
        img_path = join(self.data_dir, self.img_name_list[index])
        img = np.array(cv2.imread(img_path))
        if(self.split != 'test'):
            label_path = img_path.replace('img','mask')
            label = np.array(cv2.imread(label_path,-1))

        appearance = np.zeros(14,dtype=bool) 
        appearance[np.unique(label)] = True
        prompts = []

        if(self.split != 'test'):
            for label_id in range(0,14):
                prompt = {}
                mask = label == label_id
                pixels = mask.nonzero()
                if('point' in self.prompt_class):
                    sample_num = self.point_num             
                    #sample points randomly in gt area
                    if(not appearance[label_id]): #如果gt中不存在该器官
                        rand_point = [[0,0] for i in range(sample_num)]
                    else:  
                        rand_points = np.random.randint(low=0,high=pixels[0].shape[0],size=sample_num)
                        rand_point = [[pixels[0][rand_points[i]],pixels[1][rand_points[i]]] for i in range(sample_num)]
                    #set parameter fot predict
                    prompt['point_coords'] = np.array(rand_point) 
                    prompt['point_labels'] = np.ones(sample_num)
                    # print(prompts['point_coords'])
                if('box' in self.prompt_class):
                    if(not appearance[label_id]):
                        box = np.array([0,0,0,0])
                    else:
                        x1 = pixels[0].min()
                        x2 = pixels[0].max()
                        y1 = pixels[1].min()
                        y2 = pixels[1].max()
                    box = np.array([x1, y1, x2, y2])
                    prompt['box'] = box
                
                prompts.append(prompt)
                
            batch_prompts = {}
            for oid,prompt in enumerate(prompts):
                for key in prompt.keys():
                    if(key not in batch_prompts.keys()):
                        batch_prompts[key] = prompt[key][None]
                    else:
                        batch_prompts[key] = np.concatenate([batch_prompts[key], prompt[key][None]], axis=0)
            
            if(not self.need_backgrond):
                appearance[0] = False
            
            return img, self.get_embedding(self.img_name_list[index]), label, prompts, batch_prompts, appearance

        else:
            return img, self.get_embedding(self.img_name_list[index]), None, None, None, None


def btcv_collate_fn(batch):
    '''
    :param batch: list[img,label,prompts,appearance]
    :return img: np.ndarray(B,H,W,C)
    :return label: torch.Tensor(B,H,W)
    :return prompts: list[dict{},dict{}...].每个dict代表图片中一个器官的prompt,dict的格式参考criterion.py的第76行
    :return appearance: torch.Tensor(B,13,dtype=bool). 图片中是否存在每个器官的gt
    '''
    img, embedding, gt_label, prompts, batch_prompts, appearance = zip(*batch)
    img = np.stack(img, axis=0)
    gt_label = torch.from_numpy(np.stack(gt_label, axis=0))
    appearance = np.stack(appearance, axis=0)
    # print(img, embedding, gt_label, prompts, batch_prompts, appearance)
    return img, embedding, gt_label, prompts, batch_prompts, appearance
