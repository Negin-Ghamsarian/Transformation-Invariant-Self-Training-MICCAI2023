import segmentation_models_pytorch as smp
import torch
from torchsummary import summary
import torch.nn as nn

def init_weights(module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_() 


class PSPNet(nn.Module):
    def __init__(self, in_channels = 3, classes = 11, pretrained=True):
        super(PSPNet, self).__init__()

        if pretrained:
            self.weights = "imagenet"
        else:
            self.weights = "None"
        
        self.model = smp.PSPNet(
            encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights = self.weights,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes)
            
        self.model.decoder.apply(init_weights)


    def forward(self, input_tensor):

        print("ALERT ALERT ALERT")
        print("THIS NETWORK HAS A PROBLEM WITH THE NIMBER OF TRAINABLE PARAMETERS")

        segmentation_head = self.model(input_tensor)

        return segmentation_head


if __name__ == '__main__':
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')


    assert str(device) == 'cuda', "THIS MODEL ONLY WORKS ON GPU!!!"


    model = smp.PSPNet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=11)
    mask = model.encoder(torch.ones([2, 3, 512, 512]))

    model.to(device=device)
    print(summary(model.encoder, (3,512,512)))

    # print(mask[-1].shape)

    # print("Checking the whole network:")

    # model = PSPNet(in_channels = 3, classes = 11, pretrained=True)
    # model.to(device=device)

    # template = torch.ones((2, 3, 512, 512)).to(device=device)
    
    # y1 = model(template)
    # print(y1.shape)
    # print(summary(model, (3,512,512)))

