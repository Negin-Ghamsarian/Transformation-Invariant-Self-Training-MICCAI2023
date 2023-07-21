#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:24:50 2022

@author: negin
"""

import torch.nn.functional as F
import torchvision.models as models
from .unet_parts import *
from torchsummary import summary
import torch.nn as nn
import torch
from torchvision.models import resnet34



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



class UNetPP_DS_Res34(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetPP_DS_Res34, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Backbone = Res34_Separate_org()       

        self.up_XS0 = Up(64, 64, 32, bilinear)
        
        self.up_S1 = Up(128, 64, 64, bilinear)
        self.up_S0 = Up(64, 32, 32, bilinear)
        
        self.up_L2 = Up(256, 128, 128, bilinear)
        self.up_L1 = Up(128, 64, 64, bilinear)
        self.up_L0 = Up(64, 32, 32, bilinear)
        
        self.up_XL3 = Up(512, 256, 256, bilinear)
        self.up_XL2 = Up(256, 128, 128, bilinear)
        self.up_XL1 = Up(128, 64, 64, bilinear)
        self.up_XL0 = Up(64, 32, 32, bilinear)
        

        self.outc_XL = DecoderBlock(32, n_classes)    

        self.outc_L = DecoderBlock(32, n_classes)
        self.outc_S = DecoderBlock(32, n_classes)
        self.outc_XS = DecoderBlock(32, n_classes)

    def forward(self, x):
        
        out1, out2, out3, out4, out5 = self.Backbone(x)      
        
        XS0 = self.up_XS0(out2, out1)
        
        S1 = self.up_S1(out3, out2)
        S0 = self.up_S0(S1, XS0)
        
        L2 = self.up_L2(out4, out3)
        L1 = self.up_L1(L2, S1)
        L0 = self.up_L0(L1, S0)
        
        XL3 = self.up_XL3(out5, out4)
        XL2 = self.up_XL2(XL3, L2)
        XL1 = self.up_XL1(XL2, L1)
        XL0 = self.up_XL0(XL1, L0)
       
        logits = self.outc_XL(XL0)
        L = self.outc_L(L0)
        S = self.outc_S(S0)
        XS = self.outc_XS(XS0)
        
        return logits, L, S, XS

    
    
    
if __name__ == '__main__':
    
    model = UNetPP_DS_Res34(n_channels=3, n_classes=1, bilinear=False)
    print(summary(model, (3, 512, 512)))
    template = torch.ones((1, 3, 512, 512))
    detection= torch.ones((1, 1, 512, 512))

    y1, y2, y3, y4 = model(template)
    print(y1.shape) #[1, 10, 17, 17]
    #print(y2.shape) #[1, 20, 17, 17]15    
