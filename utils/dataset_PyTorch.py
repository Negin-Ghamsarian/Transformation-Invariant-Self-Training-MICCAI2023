#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:20:54 2022

@author: negin
"""

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from .Transforms import *
from torchvision.io import read_image



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, size = (512, 512), 
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 scale=1, mask_suffix='', doTransform = True):
        
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.size = size
        self.device = device
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.mask_suffix = mask_suffix
        self.doTransform = doTransform
        
        

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        print(self.ids)            

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
        
        self.transforms = Compose([Resize(self.size),
                                   RandomApply_Customized([
                                   RandomResizedCrop(self.size, scale = (0.2, 1.0)),
                                   RandomRotation(degree=360),
                                   ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0),
                                   GaussianBlur(kernel_size=(5,5)),
                                   RandomAdjustSharpness(sharpness_factor=2)
                                   ], p = 0.5)
                                   ])
        
        
        self.transforms_necessary = Compose([Resize(self.size),
                                   #RandomApply_Customized([
                                   #RandomResizedCrop(self.size, scale = (0.2, 1.0)),
                                   #RandomRotation(degree=360),
                                   #ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5))
                                   #], p = 0.5)
                                   ])

        print("Here inside dataset")

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, i):
        idx = self.ids[i]
        
        mask_file = glob(self.masks_dir + '/' + idx + '.*')
        img_file = glob(self.imgs_dir + '/' + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
            
        img = (read_image(img_file[0])/255).type(torch.FloatTensor)    
        mask = (read_image(mask_file[0])/255).type(torch.FloatTensor)     
        
        img = img.to(self.device)
        mask = mask.to(self.device)
        

        assert img.size(dim = 1)== mask.size(dim = 1), \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
            
        if mask.size(dim = (0)) > 1:
            mask = mask[0,:,:].unsqueeze(0)
   

        if self.doTransform:
            img, mask = self.transforms(img, mask)
        else:
            img, mask = self.transforms_necessary(img, mask)



        
        
        return {
            'image': img.type(torch.cuda.FloatTensor),
            'mask': mask.type(torch.cuda.FloatTensor),
            'name': str(self.ids[i])
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
