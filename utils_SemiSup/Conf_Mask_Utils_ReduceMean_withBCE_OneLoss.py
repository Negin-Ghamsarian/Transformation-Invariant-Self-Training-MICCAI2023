#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 02:23:43 2022

@author: negin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# def Confidence_Mask(a, b, thr):
    
#     a1 = a.clone().detach()   
#     b1 = b.clone().detach()
    
#     a11 = (a1>thr).float() + (a1<(1-thr)).float()
#     b11 = (b1>thr).float() + (b1<(1-thr)).float()
    
#     Conf_Mask = torch.mul(a11, b11)
    
#     return Conf_Mask

def Confidence_Mask(a, b, thr):
    
    a1 = a.clone().detach()   
    b1 = b.clone().detach()
    
    Conf_Mask = torch.mul((a1>thr).float(), (b1>thr).float())+ torch.mul((a1<(1-thr)).float(),(b1<(1-thr)).float())
    
    return Conf_Mask

def target_computing(m1, m2):
    
    m1 = (m1 > 0.5).float()
    m2 = (m2 > 0.5).float()
    
    return m1, m2


class DiceLoss_Conf(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_Conf, self).__init__()

    def forward(self, inputs, targets, mask, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)      
        
        inputs = torch.mul(inputs, mask)
        targets = torch.mul(targets, mask)
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        dice = (2*intersection + smooth)/(total + smooth)
        
        return torch.log(torch.mean(dice))



class BCELoss_Conf(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss_Conf, self).__init__()
    def forward(self, inputs, targets, mask):    
        BCE_mean = F.binary_cross_entropy(inputs*mask, targets*mask, reduction='mean')
        return BCE_mean    

class Ens_loss(nn.Module):
    def __init__(self, thr = 0.85, weight=None, size_average=True):
        super(Ens_loss, self).__init__()
        
        self.thr = thr
        self.Dice_Conf = DiceLoss_Conf()
        
        self.BCE1 = BCELoss_Conf()
        
        
    def forward(self, inp1, inp2):
        
        inp1 = torch.sigmoid(inp1)
        inp2 = torch.sigmoid(inp2)
        
        Conf_mask = Confidence_Mask(inp1, inp2, self.thr)
        
        tar1 = inp1.clone().detach()   
        tar2 = inp2.clone().detach()
        
        tar1, tar2 = target_computing(tar1, tar2)
         
        
        loss_Dice = self.Dice_Conf(inp1, tar1, Conf_mask)
        loss_BCE = self.BCE1(inp1, tar1, Conf_mask)
        
        
        return loss_BCE







