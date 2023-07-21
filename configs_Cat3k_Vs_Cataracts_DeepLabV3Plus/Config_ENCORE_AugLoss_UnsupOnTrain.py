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
from nets_SMP import DeepLabV3Plus_Res50 as Net

Net1 = Net

Categories = ['Cat3KVsCATARACTS_0','Cat3KVsCATARACTS_1','Cat3KVsCATARACTS_2','Cat3KVsCATARACTS_3']
Learning_Rates_init = [0.001]
epochs = 100
batch_size = 4
size = (512, 512)

Dataset_Path_Train = '/storage/homefs/ng22l920/Codes/Semi_Supervised_ENCORE/Semi_Supervised_ENCORE_MICCAI23/TrainIDs_CATARACTS/'
Dataset_Path_SemiTrain = ''
Dataset_Path_Test = ''

mask_folder = ''
Results_path = '../DA_ENCORE_MICCAI23_Results/'
Visualization_path = 'visualization_Cat3kToCaDIS_UnsupOnTrain/'
Checkpoint_path = 'checkpoints_Cat3kToCaDIS_UnsupOnTrain/'
CSV_path = 'CSVs_Cat3kToCaDIS/'
project_name = "ENCORE_Cat3kToCaDIS_MIC23"

hard_label_thr = 0.75

# Warning: if the model weights are loaded, the learning rate should also change based on the number of epochs
load = False
load_path = ''#'../Semi_Supervised_ENCORE_Results1/checkpoints_Endo/Supervised'
load_epoch = ''#10


net_name = 'DeepLabV3Plus_Res50__'
strategy = "ENCORE_AugLoss_UnsupOnTrain"
test_per_epoch = 2


ensemble_batch_size = 4
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