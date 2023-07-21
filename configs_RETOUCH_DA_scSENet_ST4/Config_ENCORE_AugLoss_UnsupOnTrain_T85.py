#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:01:40 2022

@author: negin
"""

#############################################################
# Importing from a sibling directory:
import sys
sys.path.append("..")
#############################################################

from utils.Transforms import *
from nets import scSE_Net_VGG16 as Net

Net1 = Net

Categories = ['SpectralisVsTopcon4_0','SpectralisVsTopcon4_1','SpectralisVsTopcon4_2','SpectralisVsTopcon4_3']
Learning_Rates_init = [0.001]
epochs = 100
batch_size = 2
size = 'Determined in dataset method'

Dataset_Path_Train = '/storage/homefs/ng22l920/Codes/Semi_Supervised_ENCORE/Semi_Supervised_ENCORE_MICCAI23/TrainIDs_RETOUCH_DA/'
Dataset_Path_SemiTrain = ''
Dataset_Path_Test = ''
mask_folder = ''
Results_path = '../DA_ENCORE_RETOUCH_MICCAI23_Results/'
Visualization_path = 'visualization_RETOUCH/'
Checkpoint_path = 'checkpoints_RETOUCH/'
CSV_path = 'CSVs_RETOUCH/'
project_name = "ENCORE_RETOUCH_DA_MIC23_ST4"

hard_label_thr = 0.85

# Warning: if the model weights are loaded, the learning rate should also change based on the number of epochs
load = False
load_path = ''#'../Semi_Supervised_ENCORE_Results1/checkpoints_Endo/Supervised'
load_epoch = ''#10


net_name = 'scSE_Net__'
strategy = "ENCORE_AugLoss_UnsupOnTrain"
test_per_epoch = 2


ensemble_batch_size = 2
SemiSup_initial_epoch = 0
supervised_share = 1



# image_transforms = Compose([RandomApply_Customized([
#                                    ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0),
#                                    GaussianBlur(kernel_size=(5,5)),
#                                    RandomAdjustSharpness(sharpness_factor=2)
#                                    ], p = 0.5)
#                                    ])

# image_transforms = Compose([RandomApply_Customized([
#                                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
#                                    One_Of([
#                                    GaussianBlur(kernel_size=(5,5)),
#                                    RandomAdjustSharpness(sharpness_factor=1.5)
#                                    ])
#                                    ], p = 0.8)
#                                    ])

image_transforms = Compose([RandomApply_Customized([
                                   ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0),
                                   One_Of([
                                   GaussianBlur(kernel_size=(5,5)),
                                   RandomAdjustSharpness(sharpness_factor=1.5)
                                   ])
                                   ], p = 1)
                                   ])

affine = False
affine_transforms = Compose([RandomApply_Customized([
                                   RandomResizedCrop(size, scale = (0.2, 1.0)),
                                   RandomRotation(degree=360)
                                   ], p = 0.5)
                                   ])


# Unsupervised loss-weightening function parameters:  
LW = 1
GCC = 2

# blanks:
EMA_decay = ''
Alpha = ''