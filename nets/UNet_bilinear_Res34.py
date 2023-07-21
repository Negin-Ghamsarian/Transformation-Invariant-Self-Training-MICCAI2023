#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:49:46 2022

@author: negin
"""



import torch.nn.functional as F

from .unet_parts_UNet import *
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet34


class ConvTr(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.Deconv = nn.Sequential(nn.ConvTranspose2d(in_channels , out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=False))
        
    def forward(self, x):
        return self.Deconv(x)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.decode = nn.Sequential(
                                    ConvTr(in_channels, in_channels//4),
                                    OutConv(in_channels//4, out_channels))   

    def forward(self, x):
        return self.decode(x)  
    
class Res34_Separate_org(nn.Module):
    def __init__(self, pretrained=True):
        super(Res34_Separate_org,self).__init__()
        resnet = resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
                
                
    def forward(self,x):

        x = self.firstconv(x)
        x = self.firstbn(x)
        c1 = self.firstrelu(x)#1/2  64
        x = self.firstmaxpool(c1)
        
        c2 = self.encoder1(x)#1/4   64
        c3 = self.encoder2(c2)#1/8   128
        c4 = self.encoder3(c3)#1/16   256
        c5 = self.encoder4(c4)#1/32   512
        
        return c1, c2, c3, c4, c5
    
    
class UNet_Res34(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Res34, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        self.backbone = Res34_Separate_org(pretrained=True)
        self.up1 = Up(512+256, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(128, 32, bilinear)
        self.outc = DecoderBlock(32, n_classes)

    def forward(self, x):
        
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    
if __name__ == '__main__':
    model = UNet_Res34(n_channels=3, n_classes=1)
    summary(model, (3, 512, 512))
    template = torch.ones((1, 3, 512, 512))
    detection= torch.ones((1, 1, 512, 512))

    y1 = model(template)
    print(y1.shape)    