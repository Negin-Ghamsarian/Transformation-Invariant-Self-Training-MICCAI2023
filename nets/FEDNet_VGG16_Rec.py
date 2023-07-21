#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:48:36 2022

@author: negin
"""


import torch
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
from torchvision.models import resnet34
from .unet_parts import *

class Rec_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2, out_channels) 
        
    def forward(self,x):
        y = self.conv(self.up(x))
        
        return y
           
        
class Decoder_Unsup(nn.Module):
    def __init__(self, n_channels):
        super(Decoder_Unsup, self).__init__()
        self.up1 = Rec_Up(n_channels, 256)
        self.up2 = Rec_Up(256, 128)
        self.up3 = Rec_Up(128, 64)
        self.up4 = Rec_Up(64, 3)
        
    def forward(self,x):
        out1 = self.up1(x)
        out2 = self.up2(out1)
        out3 = self.up3(out2)
        out4 = self.up4(out3)
         
        return out4

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

        return out2, out3, out4, out5


class ResConv(nn.Module):
    """(convolution => ReLU) ++"""

    def __init__(self, in_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.double_conv(x) + x
        return y




class DecoderBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels , out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  
        )

    def forward(self, x):
        return self.deconv(x)





class DensUpConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.double_conv(x)





class SE_block(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.global_pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid()
                )
        
    def forward(self, x):
        return self.global_pooling(x)                



class ConvUp(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, scale):
        super().__init__()

        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.conv_up(x)


class FeatureFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.SE_low = SE_block(channels)
        self.SE_high = SE_block(channels)

    def forward(self, low, high):
        out = low + high + self.SE_low(low) + self.SE_high(high)
        return out





class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x) 

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

class FEDNet_VGG16_Rec(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, pretrained=True):
        super(FEDNet_VGG16_Rec, self).__init__()
        assert n_channels == 3
        self.w = 512
        self.h = 512
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.backbone = VGG_Separate()
        
        self.RCB1 = ResConv(512)
        self.RCB2 = ResConv(512)
        self.RCB3 = ResConv(256)
        self.RCB4 = ResConv(128)
        
        self.DUC = DensUpConv(512, 512)
        
        self.CU12 = ConvUp(512, 512, 2)
        self.CU13 = ConvUp(512, 256, 4)
        self.CU14 = ConvUp(512, 128, 8)
        
        self.CU23 = ConvUp(512, 256, 2)
        self.CU24 = ConvUp(512, 128, 4)
        
        self.CU34 = ConvUp(256, 128, 2)

        
        self.FF2 = FeatureFusion(512)
        self.FF3 = FeatureFusion(256)
        self.FF4 = FeatureFusion(128)
        
        
        self.Dec2 = DecoderBlock(512, 256)
        self.Dec3 = DecoderBlock(256, 128)
        self.Dec4 = DecoderBlock(128, 64)
               
        
        self.outc = OutConv(64,1)     
        
        self.Rec_Branch = Decoder_Unsup(512)
        
    def forward(self, x, y):
        
        F4, F3, F2, F1 = self.backbone(x)
        
        
        Res1 = self.RCB1(F1)

        
        Res2 = self.RCB2(F2)
        Res3 = self.RCB3(F3)
        Res4 = self.RCB4(F4)
        
        
        up1 = self.DUC(Res1)
        
        cu12 = self.CU12(Res1)
        cu13 = self.CU13(Res1)
        cu14 = self.CU14(Res1)
        
        cu23 = self.CU23(Res2)
        cu24 = self.CU24(Res2)
        
        cu34 = self.CU34(Res3)
        
        fuse2 = self.FF2(cu12, Res2)
        fuse3 = self.FF3(cu13+cu23, Res3)
        fuse4 = self.FF4(cu14+cu24+cu34, Res4)
        
        add2 = fuse2 + up1
        decode2 = self.Dec2(add2)

        
        add3 = fuse3 + decode2
        decode3 = self.Dec3(add3)
        
        add4 = fuse4 + decode3
        decode4 = self.Dec4(add4)
        out = self.outc(decode4)
        
        rec2, rec3, rec4, rec5 = self.backbone(y) 
        #print(rec1.shape)
        rec_f = self.Rec_Branch(rec5)
        
        return out, rec_f
        
        
        
class FEDNet_light(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, pretrained=True):
        super(FEDNet_light, self).__init__()
        assert n_channels == 3
        self.w = 512
        self.h = 512
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.backbone = VGG_Separate()
        
        self.RCB1 = ResConv(1024)
        self.RCB2 = ResConv(512)
        self.RCB3 = ResConv(256)
        self.RCB4 = ResConv(64)
        
        self.DUC = DensUpConv(1024, 512)
        
        self.CU12 = ConvUp(1024, 512, 2)
        self.CU13 = ConvUp(1024, 256, 4)
        self.CU14 = ConvUp(1024, 64, 8)
        
        self.CU23 = ConvUp(512, 256, 2)
        self.CU24 = ConvUp(512, 64, 4)
        
        self.CU34 = ConvUp(256, 64, 2)

        
        self.FF2 = FeatureFusion(512)
        self.FF3 = FeatureFusion(256)
        self.FF4 = FeatureFusion(64)
        
        
        self.Dec2 = DecoderBlock(512, 256)
        self.Dec3 = DecoderBlock(256, 64)
        self.Dec4 = DecoderBlock(64, 32)
               
        
        self.outc = OutConv(32)     
        
    def forward(self, x):
        
        F4, F3, F2, F1 = self.backbone(x)

        
        Res1 = self.RCB1(F1)
        Res2 = self.RCB2(F2)
        Res3 = self.RCB3(F3)
        Res4 = self.RCB4(F4)
        
        
        up1 = self.DUC(Res1)
        
        cu12 = self.CU12(Res1)
        cu13 = self.CU13(Res1)
        cu14 = self.CU14(Res1)
        
        cu23 = self.CU23(Res2)
        cu24 = self.CU24(Res2)
        
        cu34 = self.CU34(Res3)
        
        fuse2 = self.FF2(cu12, Res2)
        fuse3 = self.FF3(cu13+cu23, Res3)
        fuse4 = self.FF4(cu14+cu24+cu34, Res4)
        
        add2 = fuse2 + up1
        decode2 = self.Dec2(add2)

        
        add3 = fuse3 + decode2
        decode3 = self.Dec3(add3)
        
        add4 = fuse4 + decode3
        decode4 = self.Dec4(add4)
        out = self.outc(decode4)
        
        return out        
        
        














if __name__ == '__main__':
    # model = Res50_backbone()
    # model1 = Res50_Separate2(pretrained=True)
    model = FEDNet_VGG16_Rec()
    #model = Res50_backbone(n_channels=3, n_classes=1, bilinear=False)

    summary(model, [(3, 512, 512), (3, 512, 512)])

    template = torch.ones((1, 3, 512, 512))
    template1 = torch.ones((1, 3, 512, 512))
    

    y1, y2 = model(template, template1)
    print(y2.shape) #[1, 10, 17, 17]
   









