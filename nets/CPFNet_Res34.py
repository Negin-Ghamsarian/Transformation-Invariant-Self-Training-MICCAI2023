# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:57:41 2021

@author: Negin
"""

import torch
from torchvision.models import resnet34
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''
class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []



class Res34_Separate(nn.Module):
    def __init__(self, pretrained):
        super(Res34_Separate,self).__init__()
        model1 = resnet34(pretrained=pretrained)
        self.model = nn.Sequential(*list(model1.children())[0:8])

        self.save_output = SaveOutput()

        self.hook_handles = []

        for layer in self.model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(self.save_output)
                self.hook_handles.append(handle)
               
                
    def forward(self,x):

        out = self.model(x)
        #print(out.shape)
        out0 = self.save_output.outputs[0]
        out1 = self.save_output.outputs[3] 
        out2 = self.save_output.outputs[9] 
        out3 = self.save_output.outputs[17]
        #out4 = self.save_output.outputs[29]
        
        return out0, out1, out2, out3, out

'''

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



class GPG2(nn.Module):
    def __init__(self, c_ins, c_out, scales):
        super().__init__()
        
        self.conv0 = ConvBnRelu(c_ins[0], c_out, 3, 1, 1, 1)
        self.conv1 = nn.Sequential(ConvBnRelu(c_ins[1], c_out, 3, 1, 1, 1),
                                   nn.UpsamplingBilinear2d(scale_factor=scales[1]))
        
        self.diconv1 = Separable_sub(2*c_out, c_out, 1)
        self.diconv2 = Separable_sub(2*c_out, c_out, 2)
        self.diconv3 = Separable_sub(2*c_out, c_out, 4)
        
        self.conv4 = ConvBnRelu(3*c_out, c_out   , 1, 1, 0, 1, 1, has_relu=False)
        
    def forward(self,x0, x1):
         
        y1 = self.conv0(x0)
        y2 = self.conv1(x1)        
        y4 = torch.cat([y1,y2], dim = 1)
        
        y5 = self.diconv1(y4)
        y6 = self.diconv2(y4)
        y7 = self.diconv3(y4)
        y8 = torch.cat([y5,y6,y7], dim = 1)

        return self.conv4(y8)




class GPG3(nn.Module):
    def __init__(self, c_ins, c_out, scales):
        super().__init__()
        
        self.conv0 = ConvBnRelu(c_ins[0], c_out, 3, 1, 1, 1)
        self.conv1 = nn.Sequential(ConvBnRelu(c_ins[1], c_out, 3, 1, 1, 1),
                                   nn.UpsamplingBilinear2d(scale_factor=scales[1]))
        self.conv2 = nn.Sequential(ConvBnRelu(c_ins[2], c_out, 3, 1, 1, 1),
                                   nn.UpsamplingBilinear2d(scale_factor=scales[2]))
        
        self.diconv1 = Separable_sub(3*c_out, c_out, 1)
        self.diconv2 = Separable_sub(3*c_out, c_out, 2)
        self.diconv3 = Separable_sub(3*c_out, c_out, 4)
        
        self.conv4 = ConvBnRelu(3*c_out, c_out   , 1, 1, 0, 1, 1, has_relu=False)
        
    def forward(self,x0, x1, x2):
         
        y1 = self.conv0(x0)
        y2 = self.conv1(x1)
        y3 = self.conv2(x2)         
        y4 = torch.cat([y1,y2,y3], dim = 1)
        
        y5 = self.diconv1(y4)
        y6 = self.diconv2(y4)
        y7 = self.diconv3(y4)
        y8 = torch.cat([y5,y6,y7], dim = 1)

        return self.conv4(y8)
    

class GPG4(nn.Module):
    def __init__(self, c_ins, c_out, scales):
        super().__init__()
        
        self.conv0 = ConvBnRelu(c_ins[0], c_out, 3, 1, 1, 1)
        self.conv1 = nn.Sequential(ConvBnRelu(c_ins[1], c_out, 3, 1, 1, 1),
                                   nn.UpsamplingBilinear2d(scale_factor=scales[1]))
        self.conv2 = nn.Sequential(ConvBnRelu(c_ins[2], c_out, 3, 1, 1, 1),
                                   nn.UpsamplingBilinear2d(scale_factor=scales[2]))
        self.conv3 = nn.Sequential(ConvBnRelu(c_ins[3], c_out, 3, 1, 1, 1),
                                   nn.UpsamplingBilinear2d(scale_factor=scales[3]))
        
        self.diconv1 = Separable_sub(4*c_out, c_out, 1)
        self.diconv2 = Separable_sub(4*c_out, c_out, 2)
        self.diconv3 = Separable_sub(4*c_out, c_out, 4)
        
        self.conv4 = ConvBnRelu(3*c_out, c_out   , 1, 1, 0, 1, 1, has_relu=False)
        
    def forward(self,x0, x1, x2, x3):
         
        y0 = self.conv0(x0)
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)  
        y3 = self.conv3(x3)  
        y4 = torch.cat([y0,y1,y2,y3], dim = 1)
        
        y5 = self.diconv1(y4)
        y6 = self.diconv2(y4)
        y7 = self.diconv3(y4)
        y8 = torch.cat([y5,y6,y7], dim = 1)

        return self.conv4(y8)


         
class Separable_sub(nn.Module):
    def __init__(self, c, c_out, dilate):
        super().__init__()
        
        self.sep = nn.Sequential(nn.Conv2d(c, c, 3, groups=c, dilation=dilate, padding=dilate),
                                 nn.BatchNorm2d(c),
                                 nn.Conv2d(c, c_out, 1),
                                 nn.BatchNorm2d(c_out),
                                 nn.ReLU(inplace=False))
        
    def forward(self,x):
        
        return self.sep(x)
        


        
class SAPF(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.SA1 = Scale_Aware(c)
        self.SA2 = Scale_Aware(c)
        self.alpha = nn.Parameter(torch.rand(1))
        
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)
        self.bn3 = nn.BatchNorm2d(c)
        
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self,x):
        
        y1 = self.bn1(self.conv1(x))
        weight = self.conv1.weight
        y2 = self.bn2(F.conv2d(x, weight, dilation=2, padding=2))
        y3 = self.bn3(F.conv2d(x, weight, dilation=4, padding=4))
        y4 = self.SA1(y1,y2)
        y5 = self.SA2(y3,y4)
        
        yf = self.relu(self.alpha*y5 + (1-self.alpha)*x)
        
        return yf
        
        
      
        
        
class Scale_Aware(nn.Module):
    def __init__(self, c):   
        super().__init__()
        
        self.conv1 = ConvBnRelu(2*c, c   , 1, 1, 0, 1, 1, has_bn=False)
        self.conv2 = ConvBnRelu(c  , c//2, 3, 1, 1, 1, 1, has_bn=False)
        self.conv3 = nn.Conv2d(c//2, 2, 3, padding=1)
        self.soft = nn.Softmax(dim=1)
        
        
    def forward(self, FA, FB):
        
        CAT = torch.cat([FA, FB], dim = 1)
        z1 = self.conv3(self.conv2(self.conv1(CAT)))
        z2 = self.soft(z1)
        z3 = torch.mul(FA, z2[:,0,:,:].unsqueeze(-3))
        z4 = torch.mul(FB, z2[:,1,:,:].unsqueeze(-3))
        z5 = z3 + z4
        
        return z5
        
        
    
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, pad, dilation=1,
                 groups=1, has_conv=True, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=False, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.has_conv = has_conv
        if self.has_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                                  stride=stride, padding=pad,
                                  dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        
        if self.has_conv:
            x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x          



class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(ConvBnRelu(in_channels, in_channels, 3, 1, 1),
                                nn.UpsamplingBilinear2d(scale_factor=2),  
                                ConvBnRelu(in_channels, out_channels, 1, 1, 0))
    def forward(self,x):
        
        return self.up(x)



class CPFNet_Res34(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, pretrained=True):
        super(CPFNet_Res34, self).__init__()
        self.w = 512
        self.h = 512
        self.n_classes = n_classes
        self.n_channels = n_channels
        
        self.backbone = Res34_Separate_org(pretrained=pretrained)
        
        self.SAP = SAPF(512)
        
        
        self.GPG_2 = GPG2([256, 512], 256, [1, 2])
        self.GPG_3 = GPG3([128, 256, 512], 128, [1, 2, 4])
        self.GPG_4 = GPG4([64, 128, 256, 512], 64, [1, 2, 4, 8])
        
        
        self.dec1 = decoder_block(512, 256)
        self.dec2 = decoder_block(256, 128)
        self.dec3 = decoder_block(128, 64)
        self.dec4 = decoder_block(64, 64)
        self.dec5 = decoder_block(64, n_classes)
        

    def forward(self,x):
        
        y0, y1, y2, y3, y4 = self.backbone(x)
        
        
        gpg2_out = self.GPG_2(y3, y4)
        gpg3_out = self.GPG_3(y2, y3, y4)
        gpg4_out = self.GPG_4(y1, y2, y3, y4)
               
        z1 = self.dec1(self.SAP(y4)) + gpg2_out
        z2 = self.dec2(z1) + gpg3_out
        z3 = self.dec3(z2) + gpg4_out
        z4 = self.dec4(z3) + y0
        z5 = self.dec5(z4)
        
        
        
        return z5
        
      
if __name__ == '__main__':
    X = torch.rand((1, 3, 512, 512))
    '''model = Res34_Separate(pretrained=True)     
    y0, y1,y2,y3,y4 = model(X)
    print(y0.shape, y1.shape, y2.shape, y3.shape, y4.shape)'''
    
    '''model = Res34_Separate_org()
    y1, y2, y3, y4, y5 = model(X)
    print(y5.shape)'''
    model = CPFNet_Res34()
    print(summary(model, (3, 512, 512)))
    '''x = torch.rand(12, 20, 25, 25)
    
    model = DAM(20,2)
    y = model(x)
    print(y.shape)'''
    #print(y.shape)        
        