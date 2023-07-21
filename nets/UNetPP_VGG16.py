# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:13:11 2020

@author: Negin Ghamsarian
"""
import torch.nn.functional as F
import torchvision.models as models
from .unet_parts import *
from torchsummary import summary
import torch.nn as nn
import torch




class VGG16_Separate(nn.Module):
    def __init__(self):
        super(VGG16_Separate,self).__init__()
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
        
                



class UNetPP_VGG16(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetPP_VGG16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Backbone = VGG16_Separate()       

        self.up_XS0 = Up(128, 64, 32, bilinear)
        
        self.up_S1 = Up(256, 128, 64, bilinear)
        self.up_S0 = Up(64, 32, 32, bilinear)
        
        self.up_L2 = Up(512, 256, 128, bilinear)
        self.up_L1 = Up(128, 64, 64, bilinear)
        self.up_L0 = Up(64, 32, 32, bilinear)
        
        self.up_XL3 = Up(512, 512, 256, bilinear)
        self.up_XL2 = Up(256, 128, 128, bilinear)
        self.up_XL1 = Up(128, 64, 64, bilinear)
        self.up_XL0 = Up(64, 32, 32, bilinear)
        


        self.outc = OutConv(32, n_classes)

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
       
        logits = self.outc(XL0)
        
        return logits

    
    
    
if __name__ == '__main__':
    
    model = UNetPP_VGG16(n_channels=3, n_classes=1, bilinear=False)
    #print(summary(model, (3, 512, 512)))
    
    template = torch.ones((1, 3, 512, 512))
    detection= torch.ones((1, 1, 512, 512))

    y1 = model(template)
    print(y1.shape) #[1, 10, 17, 17]
    #print(y2.shape) #[1, 20, 17, 17]15    
