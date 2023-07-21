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
import pandas as pd
from PIL import Image, ImageOps, ImageFilter



class BasicDataset(Dataset):
    def __init__(self, train_IDs_CSV, size = (512, 512), 
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 scale=1, mask_suffix='', doTransform = True, strategy = 'Default'):
        
        self.train_IDs_CSV = train_IDs_CSV

        if 'RETOUCH' in self.train_IDs_CSV:
            self.size = (512, 512) # min_size
            self.square_size = 900
        elif 'MRI' in self.train_IDs_CSV:
            self.size = (512, 512)  
        else:
            self.size = size
        self.device = device
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.mask_suffix = mask_suffix
        self.doTransform = doTransform
        self.strategy = strategy
        

        data = pd.read_csv(self.train_IDs_CSV, usecols = ['imgs','masks'])

        self.ids_imgs = data['imgs'].tolist()
        self.ids_masks = data['masks'].tolist()

        logging.info(f'Creating dataset with {len(self.ids_imgs)} examples')
        
        # if 'RETOUCH' in self.train_IDs_CSV:
        #     self.transforms = Compose([Resize(self.size),
        #                             RandomApply_Customized([
        #                             RandomResizedCrop(self.size, scale = (0.2, 1.0)),
        #                             RandomRotation(degree=30),
        #                             ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0),
        #                             GaussianBlur(kernel_size=(5,5)),
        #                             RandomAdjustSharpness(sharpness_factor=2)
        #                             ], p = 0.5)
        #                             # , Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #                             ])
                                        
        self.transforms = Compose([Resize(self.size),
                                RandomApply_Customized([
                                RandomResizedCrop(self.size, scale = (0.2, 1.0)),
                                RandomRotation(degree=30),
                                ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0),
                                GaussianBlur(kernel_size=(5,5)),
                                RandomAdjustSharpness(sharpness_factor=2)
                                ], p = 0.5)
                                # , Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


        self.transforms_spatial = Compose([Resize(self.size),
                                RandomApply_Customized([
                                RandomResizedCrop(self.size, scale = (0.2, 1.0)),
                                RandomRotation(degree=30)
                                ], p = 0.5)
                                # , Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])    

        self.transforms_nonspatial = Compose([
                                RandomApply_Customized([
                                ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0),
                                GaussianBlur(kernel_size=(5,5)),
                                RandomAdjustSharpness(sharpness_factor=2)
                                ], p = 0.5)
                                # , Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])                                        

        
        
        self.transforms_necessary = Compose([Resize(self.size),
                                   #RandomApply_Customized([
                                   #RandomResizedCrop(self.size, scale = (0.2, 1.0)),
                                   #RandomRotation(degree=360),
                                   #ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5))
                                   #], p = 0.5)
                                   # , Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

        print("Here inside dataset")

    def __len__(self):
        return len(self.ids_imgs)


    def __getitem__(self, i):
        idx = self.ids_imgs[i]
        
        img = (read_image(self.ids_imgs[i])/255).type(torch.FloatTensor) 

        if 'RETOUCH' in self.train_IDs_CSV or 'MRI' in self.train_IDs_CSV:
            img = img.repeat(3, 1, 1) 
            
        # .type(torch.FloatTensor)    
        mask = (read_image(self.ids_masks[i])/255).type(torch.FloatTensor)     

        if 'MRI' in self.train_IDs_CSV:
            mask = (mask == 1).float()   
        
        img = img.to(self.device)
        mask = mask.to(self.device)

        if 'RETOUCH' in self.train_IDs_CSV:
            img, mask = pad_if_smaller(img, mask, self.square_size)
        

        # print(f'torch.min(img): {torch.min(img)}')
        # print(f'torch.max(img): {torch.max(img)}')

        # print(f'torch.min(mask): {torch.min(mask)}')
        # print(f'torch.max(mask): {torch.max(mask)}')
        

        assert img.size(dim = 1)== mask.size(dim = 1), \
            f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'
            
        if mask.size(dim = (0)) > 1:
            mask = mask[0,:,:].unsqueeze(0)
   

        if self.doTransform:  
            if self.strategy != 'Default':
                img_org, mask_org = self.transforms_spatial(img, mask)
                img, mask = self.transforms_nonspatial(img_org, mask_org)

            else:
                img, mask = self.transforms(img, mask)    
        else:
            img, mask = self.transforms_necessary(img, mask)



        if self.strategy != 'Default':
            return {
            'image_org': img_org.type(torch.cuda.FloatTensor),
            'mask_org': mask_org.type(torch.cuda.FloatTensor),
            'image': img.type(torch.cuda.FloatTensor),
            'mask': mask.type(torch.cuda.FloatTensor),
            'name': str(self.ids_imgs[i])
            }
        else:    
        
            return {
                'image': img.type(torch.cuda.FloatTensor),
                'mask': mask.type(torch.cuda.FloatTensor),
                'name': str(self.ids_imgs[i])
            }

