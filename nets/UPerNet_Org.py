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


class UpConv(nn.Module):
    """up + (convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x,x1):
        x = self.up(x)

        y = self.double_conv(torch.cat([x,x1], dim = 1))
        return y


class PoolUp(nn.Module):
      def __init__(self, input_channels, pool_kernel_size, pool_stride, reduced_channels, upscale_size):
          super().__init__()

          self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride = pool_stride)
          self.conv = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, padding=0)
          self.up = nn.Upsample(scale_factor=upscale_size)
          
      def forward(self,x):
          y = self.pool(x)
          y = self.conv(y)
          y = self.up(y)
          
          return y        







class Res50_Separate(nn.Module):
    def __init__(self, pretrained=True):
        super(Res50_Separate,self).__init__()
        Res50_model = models.resnet50(pretrained=pretrained)
        self.Conv1 = nn.Sequential(*list(Res50_model.children())[0:5])
        self.Conv2 = nn.Sequential(*list(Res50_model.children())[5:6]) 
        self.Conv3 = nn.Sequential(*list(Res50_model.children())[6:7])
        self.Conv4 = nn.Sequential(*list(Res50_model.children())[7:8])


    def forward(self,x):
        out1 = self.Conv1(x)
        #print(out1.shape)
        out2 = self.Conv2(out1)
        #print(out2.shape)
        out3 = self.Conv3(out2)
        #print(out3.shape)
        out4 = self.Conv4(out3)
        #print(out4.shape)


        return out1, out2, out3, out4


class OutConv(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),            
        )

    def forward(self, x):
        return self.conv_up(x)



class PPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool1 = PoolUp(in_channels, 16, 316, out_channels, 16)
        self.pool2 = PoolUp(in_channels, 8, 8, out_channels, 8)
        self.pool3 = PoolUp(in_channels, 4, 4, out_channels, 4)
        self.pool4 = PoolUp(in_channels, 2, 2, out_channels, 2)

        #self.conv = nn.Conv2d(in_channels*3//2, in_channels//2, kernel_size=1, padding=0)

    def forward(self, x):

        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x4 = self.pool4(x)

        return torch.cat([x,x1,x2,x3,x4], dim = 1)



class Fuse(nn.Module):
    def __init__(self, up1, up2, up3, in_channels, out_channels):
        super().__init__()

        self.up1 = nn.Upsample(scale_factor=up1)
        self.up2 = nn.Upsample(scale_factor=up2)
        self.up3 = nn.Upsample(scale_factor=up3)
        

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward (self, x1, x2, x3, x4):
        
        y1 = self.up1(x1)
        y2 = self.up2(x2)
        y3 = self.up3(x3)
        

        y = torch.cat([y1,y2,y3,x4], dim = 1)

        return self.conv(y)


class UPerNet_Org(nn.Module):
      def __init__(self, n_classes=1, n_channels=3,):
          super(UPerNet_Org, self).__init__()
          

          self.n_classes = n_classes
          self.n_channels = n_channels

          self.backbone = Res50_Separate()

          self.pp = PPM(2048, 256)

          self.conv1 = UpConv(4096, 512)
          self.conv2 = UpConv(1024, 256)
          self.conv3 = UpConv(512, 64)
          

          self.fuse = Fuse(8, 4, 2, 3904, 32)
          
          self.outconv = OutConv(32,self.n_classes)
          
      def forward(self,x):
           
           f3, f2, f1, f0 = self.backbone(x)
           
           x0 = self.pp(f0)
           
           x1 = self.conv1(x0, f1)
           x2 = self.conv2(x1, f2)
           x3 = self.conv3(x2, f3)
           
           x4 = self.fuse(x0,x1,x2,x3)

           out = self.outconv(x4)
           
           return out
           
        














if __name__ == '__main__':
    model = UPerNet_Org()
    model.to(device='cuda')

    template = torch.ones((1, 3, 512, 512))

    print(summary(model, (3, 512, 512)))



    







































