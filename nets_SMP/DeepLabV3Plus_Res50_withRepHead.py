import segmentation_models_pytorch as smp
import torch.nn as nn
from torchsummary import summary
import torch

def init_weights(module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


# class DeepLabV3Plus_smp(nn.Module):
#     def __init__(self, pretrained=True):
#         super(DeepLabV3Plus_smp,self).__init__()

#     def DeepLabV3Plus_maker(self, in_channels = 3, classes = 11):

#         self.DeepLabV3Plus= smp.DeepLabV3Plus(
#             encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#             encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#             in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#             classes=classes,                      # model output channels (number of classes in your dataset)
#             )

#         return self.DeepLabV3Plus

class DeepLabV3Plus_withRepHead(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 11, pretrained=True):
        super(DeepLabV3Plus_withRepHead, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        if pretrained:
            self.weights = "imagenet"
        else:
            self.weights = "None"
        
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights = self.weights,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_classes)

        self.model.decoder.apply(init_weights)

        self.representation = nn.Sequential(
                nn.Conv2d(2048, 8, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Flatten(),
                nn.Linear(1024*8, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128)
            )

    def forward(self, input_tensor):

        segmentation_head = self.model(input_tensor)
        representation_head = self.representation(self.model.encoder(input_tensor)[-1])

        return segmentation_head, representation_head

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')


    assert str(device) == 'cuda', "THIS MODEL ONLY WORKS ON GPU!!!"

    print("THIS MODEL ONLY WORKS WITH BATCH-SIZE>1")

    # initializer = DeepLabV3Plus_smp(pretrained=True)
    # model = initializer.DeepLabV3Plus_maker()
    # model.to(device=device)
    # print(summary(model, (3,512,512)))

    # print("MODEL ARCHITECTURE")
    # print(model)

    # print("ENCODER ARCHITECTURE")
    # print(model.encoder)

    # print("DECODER ARCHITECTURE")
    # print(model.decoder)

    # print("DECODER WEIGHT INITIALIZATION")
    # model.encoder.apply(init_weights)

    # print("Summary of the model encoder: ")
    # print(summary(model.encoder, (3,512,512)))

    # template = torch.ones((2, 3, 512, 512)).to(device=device)

    # 

    # y = model(template)
    # print(y.shape)
    
    # # z1, z2, z3, z4, z5, z6 = model.encoder(template)
    # # print(z1.shape, z2.shape, z3.shape, z4.shape, z5.shape, z6.shape)




    print("Checking the whole network:")

    model = DeepLabV3Plus_withRepHead(in_channels = 3, classes = 11, pretrained=True)
    model.to(device=device)

    template = torch.ones((2, 3, 512, 512)).to(device=device)
    
    segmentation_head, representation_head = model(template)
    print(f'segmentation_head.shape: {segmentation_head.shape}')
    print(f'representation_head.shape: {representation_head.shape}')
    print(summary(model, (3,512,512)))