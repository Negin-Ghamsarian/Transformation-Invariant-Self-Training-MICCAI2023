#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 15:22:38 2022

@author: negin
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchsummary import summary

nonlinearity = partial(F.relu, inplace=False)


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
  
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)      
    
    
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, pad, dilation=1,
                 groups=1, has_relu=True, inplace=False, has_bias=False):
        super(ConvBnRelu, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x     
    
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
        
        self.decode = nn.Sequential(ConvBnRelu(in_channels, in_channels//4, 1, 1, 0),
                                    ConvTr(in_channels//4, in_channels//4),
                                    ConvBnRelu(in_channels//4, out_channels, 1, 1, 0))   

    def forward(self, x):
        return self.decode(x)
    
class DAC_Block(nn.Module):
    def __init__(self, channel):
        super(DAC_Block, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, bias=False)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3, bias=False)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0, bias=False)

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out    
        
class RMP_Block(nn.Module):
    def __init__(self, in_channels):
        super(RMP_Block, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class CE_Net_VGG16(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, pretrained=True):
        super(CE_Net_VGG16, self).__init__()
        self.w = 512
        self.h = 512
        self.n_classes = n_classes
        self.n_channels = n_channels
        
        self.backbone = VGG_Separate()
        self.DAC = DAC_Block(512)
        self.RMP = RMP_Block(512)
        
        self.decode1 = DecoderBlock(516, 512)
        self.decode2 = DecoderBlock(1024, 256)
        self.decode3 = DecoderBlock(512, 128)
        self.decode4 = DecoderBlock(256, 64)
        self.decode5 = OutConv(128, n_classes)
        
    def forward(self, x):
        
        y1, y2, y3, y4, y5 = self.backbone(x)
        
        
        y6  = self.RMP(self.DAC(y5))
        #print(y6.shape)
        
        y7  = torch.cat([self.decode1(y6), y4], dim = 1)
        y8  = torch.cat([self.decode2(y7), y3], dim = 1)
        y9  = torch.cat([self.decode3(y8), y2], dim = 1)
        y10 = torch.cat([self.decode4(y9), y1], dim = 1)
        y11 = self.decode5(y10)
        
        return y11
        
if __name__ == '__main__':
    X = torch.rand((1, 3, 512, 512))
    model = CE_Net_VGG16()
    #y = model(X)
    #print(y.shape) 
    print(summary(model, (3, 512, 512)))
