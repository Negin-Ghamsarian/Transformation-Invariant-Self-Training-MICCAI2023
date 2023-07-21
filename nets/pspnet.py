from turtle import forward
import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch
#from torchvision.ops import DeformConv2d
import torch.nn.functional as F
from torchvision.models import resnet50


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x):
        
        return self.conv(self.up(x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)       

class Res50(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, pretrained=True):
        super(Res50,self).__init__()
        resnet = resnet50(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.l1 = resnet.layer1
        self.l2 = resnet.layer2
        self.l3 = resnet.layer3
        self.l4 = resnet.layer4

        
                
                
    def forward(self,x):

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        # y4 = self.l4(y3)
        # print(f'y3.shape: {y3.shape}')
        
        return x


class PSPNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(PSPNet, self).__init__()

        self.backbone = Res50(n_channels=n_channels, n_classes=n_classes)
        self.head = PSPHead(in_channels = 1024, out_channels = n_classes)
        self.n_channels = n_channels
        self.n_classes = n_classes


    def forward(self, x):
        return self.head(self.backbone(x))


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5 = nn.Sequential(PyramidPooling(in_channels),
                                   nn.Dropout(0.2, False),
                                   Up(in_channels*2, in_channels//32),                                  
                                   OutConv(in_channels//32, out_channels))

    def forward(self, x):
        return self.conv5(x)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=True)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=True)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


if __name__ == '__main__':
    model = PSPNet(n_channels=3, n_classes=3)
    print(model)
    template = torch.ones((2, 3, 512, 512))
    #detection= torch.ones((1, 1, 512, 512))
    
    y1 = model(template)
    print('shape:', y1.shape)
    model.cuda()
    print(summary(model, (3,512,512), device = 'cuda'))        