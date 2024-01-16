import numpy as np
import torch


def dice_loss(pred, gt):
    '''
    简单定义的dice loss
    pred: (batch_size, H, W) or (H, W)
    gt: (H, W)
    '''
    smooth = 1
    intersection = torch.sum(pred * gt) + smooth  # (batch_size,)
    union = torch.sum(pred, dim=(-2,-1)) + torch.sum(gt) + smooth  # (batch_size,)
    loss = (1 - 2*intersection / union).mean()  #(,)
    return loss

