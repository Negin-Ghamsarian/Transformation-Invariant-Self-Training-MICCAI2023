# -*- coding: utf-8 -*-
"""
AdaptNet

@author: Negin Ghamsarian

"""

import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch
from torchvision.ops import DeformConv2d
import torch.nn.functional as F




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
        


class Pool_up(nn.Module):
      def __init__(self, pool_kernel_size, up_size):
          super().__init__()

          self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride = pool_kernel_size, padding = 0)
          self.up = nn.Upsample(scale_factor=up_size)
          
          
      def forward(self,x):
          y1 = self.pool(x)
          y2 = self.up(y1)
          
          
          return y1, y2


class Global_Pool_up(nn.Module):
      def __init__(self, input_size):
          super().__init__()

          self.pool = nn.AdaptiveMaxPool2d((1,1))
          self.up = nn.Upsample(scale_factor=input_size)
          
          
      def forward(self,x):
          y1 = self.pool(x)
          y2 = self.up(y1)
          
          
          return y2


class Cascade_Pooling(nn.Module):
      def __init__(self, input_channels, input_size):
          super().__init__()
          
          self.pool1 = Pool_up(2, 2)
          self.pool2 = Pool_up(2, 4)
          self.pool3 = Pool_up(2, 8)
          self.pool4 = Global_Pool_up(input_size)
          self.fc = nn.Conv2d(input_channels*5, input_channels, kernel_size=1, padding=0, groups=input_channels)
          self.conv = nn.Conv2d(input_channels, input_channels//4, kernel_size=3, padding=1)
          self.conv1 = nn.Conv2d(2*input_channels, input_channels, kernel_size=1, padding=0)
          self.out = nn.Sequential(
                   nn.LayerNorm([input_size, input_size], elementwise_affine=False),
                   nn.ReLU(inplace=True))
          
      def forward(self, x) :
          
          y1, z1 = self.pool1(x)
          y2, z2 = self.pool2(y1)
          y3, z3 = self.pool3(y2)
          z4 = self.pool4(y3)          
          
          z11 = z1.unsqueeze(2)
          z21 = z2.unsqueeze(2)
          z31 = z3.unsqueeze(2)
          z41 = z4.unsqueeze(2)
          x1 = x.unsqueeze(2)
          
          k1 = torch.cat([x1, z11, z21, z31, z41], dim = 2)
          k1 = torch.flatten(k1, start_dim=1, end_dim=2)
          k1 = self.fc(k1)          
          
          z12 = self.conv(z1)
          weights = self.conv.weight
          z22 = F.conv2d(z2, weights, padding=1)
          z32 = F.conv2d(z3, weights, padding=1)
          z42 = F.conv2d(z4, weights, padding=1)
          
          k2 = torch.cat([k1, z12, z22, z32, z42], dim = 1)
          
          k3 = self.conv1(k2)
          
          return self.out(k3)



class Up(nn.Module):

    def __init__(self, in_channels, out_channels, input_size):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Cascade_Reception(in_channels, out_channels, input_size)
            

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)            


class Cascade_Reception(nn.Module):
    def __init__(self, in_channels, out_channels, input_size):
        super().__init__()
        
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.LNR = nn.Sequential(
                   nn.LayerNorm([input_size, input_size], elementwise_affine=False),
                   nn.ReLU(inplace=True))
        self.conv_d1 = DeformConv(out_channels, out_channels, input_size)
        self.conv_d2 = DeformConv(out_channels, out_channels, input_size)
        self.conv_d3 = DeformConv(out_channels, out_channels, input_size)
        
        self.conv_share3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_share1 = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)
                      
        self.out = nn.Sequential(
                   nn.LayerNorm([input_size, input_size], elementwise_affine=False),
                   nn.ReLU(inplace=True))

        self.soft = nn.Softmax2d()

    def forward(self,x):
        
        x = self.LNR(self.conv0(x))
        y1 = self.conv_d1(x)
        y2 = self.conv_d2(y1)
        y3 = self.conv_d3(y2)
        
        y11 = self.conv_share3(y1)
        y12 = self.conv_share1(y11)
        
        weight3 = self.conv_share3.weight
        weight1 = self.conv_share1.weight
        
        y21 = F.conv2d(y2, weight3, padding=1)
        y22 = F.conv2d(y21, weight1, padding=0)
        
        y31 = F.conv2d(y3, weight3, padding=1)
        y32 = F.conv2d(y31, weight1, padding=0)
        
        
        concat = torch.cat([y12, y22, y32], dim = 1)
        soft = self.soft(concat)
        
        y11 = torch.mul(y11, soft[:,0,:,:].unsqueeze(-3))
        y21 = torch.mul(y21, soft[:,1,:,:].unsqueeze(-3))
        y31 = torch.mul(y31, soft[:,2,:,:].unsqueeze(-3))
        
        y = y11+y21+y31
        
        return self.out(y)
        
        
        
        
class Deform(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilate):
        super().__init__()
        
        self.offset = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=3, padding = 1, dilation = 1)
        self.tan = nn.Hardtanh()
        self.deform = DeformConv2d(in_channels, out_channels, kernel_size = 3, stride = (1,1), 
                                    padding = dilate, dilation = dilate)

        
    def forward(self,x):
        
        off = self.offset(x)
        off1 = self.tan(off)
        out = self.deform(x, off1)
        weights = self.deform.weight

        return out, weights


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, input_size):
        super().__init__()
           
        self.conv1 = Deform(in_channels, in_channels, kernel_size=3, dilate = 1)
        
        self.conv_share3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_share1 = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)

        self.soft = nn.Softmax2d()

    def forward(self, x):
        

        x1, weights = self.conv1(x)
        x0 = F.conv2d(x, weights, padding=1)
        
        x11 = self.conv_share3(x1)
        weight3 = self.conv_share3.weight
        x01 = F.conv2d(x0, weight3, padding=1)
        
        x12 = self.conv_share1(x11)
        weight1 = self.conv_share1.weight
        x02 = F.conv2d(x01, weight1, padding=0)
      
        concat = torch.cat([x02, x12], dim =1)
        soft = self.soft(concat)
        
        y11 = torch.mul(x11, soft[:,1,:,:].unsqueeze(-3))
        y01 = torch.mul(x01, soft[:,0,:,:].unsqueeze(-3))
        
        y = y11+y01
        
        return y
        
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)        


class AdaptNet_VGG16(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(AdaptNet_VGG16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Backbone = VGG_Separate()

        self.glob1 = Cascade_Pooling(input_channels=512, input_size=32)

        self.up1 = Up(1024, 256, 64)
        self.up2 = Up(512, 128, 128)
        self.up3 = Up(256, 64, 256)
        self.up4 = Up(128, 32, 512)

        self.outc = OutConv(32, n_classes)


    def forward(self, x):
        
        out5, out4, out3, out2, out1 = self.Backbone(x)    
        out1 = self.glob1(out1)
        
        x1 = self.up1(out1, out2)
        x2 = self.up2(x1, out3)
        x3 = self.up3(x2, out4)
        x4 = self.up4(x3, out5)
        logits = self.outc(x4)


        return logits

    
    
    
if __name__ == '__main__':
    model = AdaptNet_VGG16(n_channels=3, n_classes=1)

    template = torch.ones((1, 3, 512, 512))
    detection= torch.ones((1, 1, 512, 512))
    
    y1 = model(template)
    print(y1.shape)
    print(summary(model, (3,512,512)))
