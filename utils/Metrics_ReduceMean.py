import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange


class Dice_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice_binary, self).__init__()

    def forward(self, inputs, targets, eps = 0.0001):


        inputs = F.sigmoid(inputs)


        inputs = (inputs>0.5).int()  
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        dice = torch.mean((2*intersection + eps)/(total + eps)).detach().cpu().item()

        del inputs, intersection, total
        
        return dice
    
    
class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, eps = 0.0001):

        inputs = F.sigmoid(inputs)
     
        inputs = (inputs>0.5).int()
        
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        union = total - intersection 
        
        IoU = torch.mean((intersection + eps)/(union + eps)).detach().cpu().item()
        del inputs, total, intersection, union
        
        
                
        return IoU



class Dice_MultiClass(nn.Module):
    
    def __init__(self, num_classes, ignore_first=True, apply_softmax=True):
        super(Dice_MultiClass, self).__init__()
        self.eps = 1
        self.ignore_first = ignore_first
        self.apply_softmax = apply_softmax
        self.num_classes = num_classes


        if ignore_first:

            self.intersection = torch.zeros(num_classes-1).to(device='cuda')
            self.cardinality = torch.zeros(num_classes-1).to(device='cuda')

        else:

            self.intersection = torch.zeros(num_classes).to(device='cuda')
            self.cardinality = torch.zeros(num_classes).to(device='cuda')



    def forward(self, input, target):


        if self.apply_softmax:
            input = input.softmax(dim=1)
        input = torch.argmax(input, dim = 1).squeeze(1)



        input_one_hot = F.one_hot(input.long(), num_classes=self.num_classes).permute(0,3,1,2)
        
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0,3,1,2)

       

        if self.ignore_first:
            input_one_hot = input_one_hot[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

            
        intersection = torch.sum(input_one_hot * target_one_hot, dim=(0, 2, 3))
        cardinality = torch.sum(input_one_hot + target_one_hot, dim=(0, 2, 3))


        self.intersection += intersection
        self.cardinality += cardinality

    def evaluate(self):    

        dice = (2. * self.intersection + self.eps) / (self.cardinality + self.eps)

        
        return dice, torch.mean(dice)     



class IoU_MultiClass(nn.Module):

    def __init__(self, num_classes, ignore_first=True, apply_softmax=True):
        super(IoU_MultiClass, self).__init__()
        self.eps = 1
        self.ignore_first = ignore_first
        self.num_classes = num_classes
        self.apply_softmax = apply_softmax


        if ignore_first:

            self.intersection = torch.zeros(num_classes-1).to(device='cuda')
            self.denominator = torch.zeros(num_classes-1).to(device='cuda')

        else:

            self.intersection = torch.zeros(num_classes).to(device='cuda')
            self.denominator = torch.zeros(num_classes).to(device='cuda')

    def forward(self, input, target):

        if self.apply_softmax:
            input = input.softmax(dim=1)
        input = torch.argmax(input, dim = 1)
        input = torch.squeeze(input, 1)

        input_one_hot = F.one_hot(input.long(), num_classes=self.num_classes).permute(0,3,1,2)
        
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0,3,1,2)

        if self.ignore_first:
            input_one_hot = input_one_hot[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
            

        intersection = torch.sum(input_one_hot * target_one_hot, dim=(0, 2, 3))
        denominator = torch.sum(input_one_hot + target_one_hot, dim=(0, 2, 3)) - intersection


        self.intersection += intersection
        self.denominator += denominator


    def evaluate(self):       


        IoU = (self.intersection + self.eps) / (self.denominator + self.eps)

        return IoU, torch.mean(IoU)


