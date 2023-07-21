# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:11:18 2020

@author: Negin
"""

import torch.nn.functional as F

from .unet_parts_UNet import *
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models

class VGG_Separate(nn.Module):
    def __init__(self):
        super(VGG_Separate,self).__init__()
        
        vgg_model = models.vgg16(pretrained=True)
        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9]) 
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[16:23])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[23:30])

    def forward(self,x):
        out1 = self.Conv1(x)
        out2 = self.Conv2(out1)
        out3 = self.Conv3(out2)
        out4 = self.Conv4(out3)
        out5 = self.Conv5(out4)

        return out1, out2, out3, out4, out5

class UNet_VGG16(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_VGG16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        self.backbone = VGG_Separate()
        self.up1 = Up(1024, 512 // 2, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    
if __name__ == '__main__':
    model = UNet_VGG16(n_channels=3, n_classes=1)
    summary(model, (3, 512, 512))
    template = torch.ones((1, 3, 512, 512))
    detection= torch.ones((1, 1, 512, 512))

    y1 = model(template)
    print(y1.shape)    