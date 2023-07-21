# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:39:03 2020

@author: Negin
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# =============================================================================
# some good resources:
# https://smp.readthedocs.io/en/latest/losses.html  
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook  
# =============================================================================

# For FocalLoss:
ALPHA = 0.8
GAMMA = 2    

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # print(f'Target Min: {torch.min(targets)}')
        # print(f'Target Max: {torch.max(targets)}')
        # print(f'Input Min: {torch.min(inputs)}')
        # print(f'Input Max: {torch.max(inputs)}')

        inputs = F.sigmoid(inputs)


        # assert torch.max(inputs)>1 or torch.min(inputs)<0 or torch.max(targets)>1 or torch.min(targets)<0, \
        #     f'Inputs and targets should be in range of [0,1]'
        # inputs = (inputs>0.5).int()  
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        dice = (2*intersection + smooth)/(total + smooth)
        
        return 1 - torch.mean(dice)

# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     """Dice coeff for individual examples"""

#     def forward(self, input, target):

#         assert torch.max(input)>1 or torch.min(input)<0 or torch.max(target)>0 or torch.min(target)<0, \
#             f'Inputs and targets should be in range of [0,1]'
#         # input = torch.sigmoid(input)  # For binary classification, we should not use sigmoid
#         eps = 0.0001


#         self.inter = torch.dot(input.view(-1), target.view(-1))
#         self.union = torch.sum(input) + torch.sum(target) + eps

#         t = (2 * self.inter.float() + eps) / self.union.float()
#         return 1-t



class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        

        
        inputs = F.sigmoid(inputs)
        # assert torch.max(inputs)>1.0 or torch.min(inputs)<0.0 or torch.max(targets)>1.0 or torch.min(targets)<0.0, \
        #     f'Inputs and targets should be in range of [0,1]'
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        #Dice_BCE = BCE + dice_loss
               
        # inputs = (inputs>0.5).int() 
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        dice = (2*intersection + smooth)/(total + smooth)
        
        # print(f'Train Dice: {dice}')
        # print(f'Train BCE: {BCE}')
        Dice_BCE = 0.8*BCE - 0.2*torch.log(torch.mean(dice))
        
        return Dice_BCE
    
    
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth = 1):

        inputs = torch.sigmoid(inputs)
        
        # print(f'Target Min: {torch.min(targets)}')
        # print(f'Target Max: {torch.max(targets)}')
        # print(f'Input Min: {torch.min(inputs)}')
        # print(f'Input Max: {torch.max(inputs)}')
        
        # assert torch.max(inputs)>1 or torch.min(inputs)<0 or torch.max(targets)>1 or torch.min(targets)<0, \
        #     f'Inputs and targets should be in range of [0,1]'  
        # inputs = (inputs>0.5).int()
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        
                
        return 1 - torch.mean(IoU) 

    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss    
    
    