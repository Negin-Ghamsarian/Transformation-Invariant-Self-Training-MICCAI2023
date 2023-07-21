#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:32:06 2022

@author: negin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:12:54 2021

@author: Negin
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
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
        
        return c2, c3, c4, c5


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


    def __init__(self, in_channels):
        super().__init__()

        self.conv_up = nn.Sequential(
                nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2),
                nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),            
        )

    def forward(self, x):
        return self.conv_up(x)



class OutConv1(nn.Module):


    def __init__(self, in_channels):
        super().__init__()

        self.conv_up = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),            
        )

    def forward(self, x):
        return self.conv_up(x)



class FEDNet_Res34(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, pretrained=True):
        super(FEDNet_Res34, self).__init__()
        assert n_channels == 3
        self.w = 512
        self.h = 512
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.backbone = Res34_Separate_org()
        
        self.RCB1 = ResConv(512)
        self.RCB2 = ResConv(256)
        self.RCB3 = ResConv(128)
        self.RCB4 = ResConv(64)
        
        self.DUC = DensUpConv(512, 256)
        
        self.CU12 = ConvUp(512, 256, 2)
        self.CU13 = ConvUp(512, 128, 4)
        self.CU14 = ConvUp(512, 64, 8)
        
        self.CU23 = ConvUp(256, 128, 2)
        self.CU24 = ConvUp(256, 64, 4)
        
        self.CU34 = ConvUp(128, 64, 2)

        
        self.FF2 = FeatureFusion(256)
        self.FF3 = FeatureFusion(128)
        self.FF4 = FeatureFusion(64)
        
        
        self.Dec2 = DecoderBlock(256, 128)
        self.Dec3 = DecoderBlock(128, 64)
        self.Dec4 = DecoderBlock(64, 64)
               
        
        self.outc = OutConv(64)     
        
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
        
        
        
class FEDNet_light(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, pretrained=True):
        super(FEDNet_light, self).__init__()
        assert n_channels == 3
        self.w = 512
        self.h = 512
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.backbone = Res34_Separate_org()
        
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
               
        
        self.outc = OutConv1(32)     
        
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
    model3 = FEDNet_Res34()
    #model = Res50_backbone(n_channels=3, n_classes=1, bilinear=False)

    template = torch.ones((1, 3, 512, 512))
    detection= torch.ones((1, 1, 512, 512))

    y1 = model3(template)
    print(summary(model3, (3,512,512)))
    print(y1.shape)    
    







































