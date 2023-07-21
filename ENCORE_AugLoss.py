#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:43:53 2022

@author: negin
"""
from __future__ import print_function
import random 
import argparse
import logging 
import os 
import sys 
import csv

import numpy as np 
import torch 
import torch.nn as nn 
from torch import optim 
from tqdm import tqdm 

from utils.eval_dice_IoU_binary import eval_dice_IoU_binary
from utils.save_metrics import save_metrics


from utils.dataset_PyTorch import BasicDataset as BasicDataset
from utils.dataset_PyTorch_CSV import BasicDataset as BasicDataset_CSV
from torch.utils.data import DataLoader

from torchvision import transforms
from utils.losses_binary_ReduceMean import DiceBCELoss

from utils_SemiSup.Conf_Mask_Utils_ReduceMean_withBCE_OneLoss import Ens_loss

import wandb
from utils.seed_initialization import seed_all, seed_worker

from utils.import_helper import import_config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
import importlib

from utils.TrainUtils import create_directory, polynomial_LR


class printer(nn.Module):
        def __init__(self, global_dict=globals()):
            super(printer, self).__init__()
            self.global_dict = global_dict
            self.except_list = []
        def debug(self,expression):
            frame = sys._getframe(1)

            print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))

        def namestr(self,obj, namespace):
            return [name for name in namespace if namespace[name] is obj]     
        
        def forward(self,x):
            for i in x:
                if i not in self.except_list:
                    name = self.namestr(i, globals())
                    if len(name)>1:
                        self.except_list.append(i)
                        for j in range(len(name)):
                            self.debug(name[j])
                    else:  
                        self.debug(name[0])


           

def train_net(net,
              epochs=30,
              batch_size=1,
              lr=0.001,
              device='cuda',
              save_cp=True
              ):

    TESTS = []
    if dataset_mode == 'folder':
        train_dataset = BasicDataset(dir_train_img, dir_train_mask)
        test_dataset = BasicDataset(dir_test_img, dir_test_mask, doTransform = False)
        test_dataset_ensemble = BasicDataset(dir_SemiTrain_img, dir_SemiTrain_mask, doTransform = False)

    elif dataset_mode == 'csv':   
        train_dataset = BasicDataset_CSV(train_IDs_CSV)
        test_dataset = BasicDataset_CSV(test_IDs_CSV, doTransform = False)
        test_dataset_ensemble = BasicDataset_CSV(semi_train_IDs_CSV, doTransform = False)
        sourceTest_dataset = BasicDataset_CSV(SourceTest_IDs_CSV, doTransform = False)

    
    n_train = len(train_dataset)
    if n_train%batch_size == 1:
        drop_last = True
    else:
        drop_last = False    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=drop_last)
    
    inference_step = np.floor(np.ceil(n_train/batch_size)/test_per_epoch)
    print(f'Inference Step:{inference_step}')

    n_test_ensemble = len(test_dataset_ensemble)
    if n_test_ensemble%ensemble_batch_size == 1:
        drop_last = True
    else:
        drop_last = False  
    test_loader_ensemble = DataLoader(test_dataset_ensemble, batch_size=ensemble_batch_size, shuffle=True, pin_memory=False, drop_last=drop_last)
    test_loader_ensemble_iterator = iter(test_loader_ensemble)

    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
    n_test = len(test_dataset)

    if dataset_mode == 'csv':
        SourceTest_loader = DataLoader(sourceTest_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
        n_SourceTest = len(sourceTest_dataset)

    
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Test size:       {n_test}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.9)

    criterion = DiceBCELoss()
    criterion2 = Ens_loss(thr=hard_label_thr)
    test_counter = 1
    for epoch in range(epochs):
        net.train()
        

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                imgs = batch['image']
                true_masks = batch['mask']
                
            
                
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                
                masks_pred = net(imgs)
                loss_main = criterion(masks_pred, true_masks)
                loss_wandb = loss_main
                
                
                ########################################################################
                ########################################################################
                if epoch > SemiSup_initial_epoch-1:
                    try:
                        Unsup_batch = next(test_loader_ensemble_iterator)
                    except StopIteration:
                        test_loader_ensemble_iterator = iter(test_loader_ensemble)
                        Unsup_batch = next(test_loader_ensemble_iterator)
                    imgs_T = Unsup_batch['image']
                    imgs_T = imgs_T.to(device=device, dtype=torch.float32)
                    
                    
                    rs = np.random.randint(2147483647, size=1)
                    rs = rs[0]
                    
    
                    random.seed(rs)
                    torch.random.manual_seed(rs)
                    imgs_TA, NI = image_transforms (imgs_T, torch.zeros_like(imgs_T))
                    
                    if affine:
                        random.seed(rs)
                        torch.random.manual_seed(rs)
                        imgs_TA, NI  = affine_transforms(imgs_TA, torch.zeros_like(imgs_T))
                    
                    
                    
                    
                    masks_TA = net(imgs_TA)
                    
                    with torch.no_grad():
                        masks_T = net(imgs_T)
                    
                    if affine:
                        random.seed(rs)
                        torch.random.manual_seed(rs)
                        masks_T = affine_transforms (masks_T)
                    
                    # # check:
                    # random.seed(rs) 
                    # torch.random.manual_seed(rs)
                    # imgs_TA_test2 = affine_transforms (imgs_T)
                    
                    
                    
                    loss_T = criterion2(masks_TA, masks_T)
                    Ti = 1/(epochs-SemiSup_initial_epoch)   
                    Lambda = LW*torch.exp(torch.tensor(0-GCC*(1-(epoch-SemiSup_initial_epoch)*Ti))) 
                    
                    
                    loss_main = loss_main + Lambda*loss_T
                ########################################################################
                
        
                loss = loss_main
                epoch_loss += loss.item()

                

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
            
                (loss_main).backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                
                ########################################################################
                ########################################################################
                if (global_step) % (inference_step) == 0: # Should be changed if the condition that the n_train%batch_size != 0
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        

                    val1, val2, val3, val4, val5, val6, val7, val8, inference_time = eval_dice_IoU_binary(net, test_loader, device, test_counter, save_test, save=False)
                    
                    print(f'Validation Dice:{val1}')
                    print(f'Validation IoU:{val3}')

                    TESTS.append([val1, val2, val3, val4, val5, val6, val7, val8, inference_time, epoch_loss])

                    
                    val1_source = 0
                    val3_source = 0
                    if dataset_mode == 'csv':
                        
                        val1_source, _, val3_source, _, _, _, _, _, _ = eval_dice_IoU_binary(net, SourceTest_loader, device, test_counter, save_test, save=False)
                        
                        print(f'Source Validation Dice:{val1_source}')
                        print(f'Source Validation IoU:{val3_source}')

                    test_counter = test_counter+1
                    
                    

                    if net.n_classes > 1:
                         print("NOT IMPLEMENTED")
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val1))
                        logging.info('Validation IoU: {}'.format(val3))

                    # Important about wandb: By default, each call to wandb.log is a new step and all of our charts and panels use the history step as the x-axis.    
                        
                    wandb.log({'Train_Loss': loss_wandb,
                    'Test_Dice': val1,
                    'Test_IoU': val3,
                    'SourceTest_Dice': val1_source,
                    'SourceTest_IoU': val3_source})
                    
                   


        scheduler.step()
           
        if save_cp:
            if True:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
        if (epoch+1)%50 == 0:
            torch.save(net.state_dict(),
                    dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    val1, val2, val3, val4, val5, val6, val7, val8, inference_time = eval_dice_IoU_binary(net, test_loader, device, test_counter, save_test, save=False)
    save_metrics(TESTS, csv_name)
     

    


if __name__ == '__main__':

    args = parser.parse_args()
    config_file = args.config
    my_conf = importlib.import_module(config_file)
    Categories, Learning_Rates_init, epochs, batch_size, size,\
             Dataset_Path_Train, Dataset_Path_SemiTrain, Dataset_Path_Test,\
                  mask_folder, Results_path, Visualization_path,\
                 CSV_path, project_name, load, load_path, net_name,\
                      test_per_epoch, Checkpoint_path, Net1,\
                         hard_label_thr, ensemble_batch_size, SemiSup_initial_epoch,\
                             image_transforms, affine, affine_transforms, LW,\
                                 EMA_decay, Alpha, strategy, GCC, supervised_share\
                     = import_config.execute(my_conf)
    print("inside main")
    print('Hello Ubelix')
    print(f'Cuda Availability: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')
    print(f'Cuda Device Number: {torch.cuda.current_device()}')
    print(f'Cuda Device Name: {torch.cuda.get_device_name(0)}')
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
    
    printer1 = printer()       
    
    print('CONFIGS:________________________________________________________')
        
    printer1([Categories, Learning_Rates_init, epochs, batch_size, size,\
                Dataset_Path_Train, Dataset_Path_SemiTrain, Dataset_Path_Test,\
                    mask_folder, Results_path, Visualization_path,\
                    CSV_path, project_name, load, load_path, net_name,\
                        test_per_epoch, Checkpoint_path, Net1,\
                            hard_label_thr, ensemble_batch_size, SemiSup_initial_epoch,\
                                image_transforms, affine, affine_transforms, LW,\
                                    EMA_decay, Alpha, strategy, GCC, supervised_share])      
    
    try:
        for c in range(len(Categories)):      
            for LR in range(len(Learning_Rates_init)):

                print(f'Initializing the learning rate: {Learning_Rates_init[LR]}')

                wandb.init(project=project_name+'_'+net_name+str(Learning_Rates_init[LR])+'_'+str(batch_size), entity="negin_gh",
                name=strategy+"_init_epoch_"+str(SemiSup_initial_epoch)+'_UnsupBatchSize_'+str(ensemble_batch_size)+'_'+"_GCC_"+ str(GCC) +"_Thr_"+str(hard_label_thr)+ '_supervised_share_'+ str(supervised_share),
                reinit=False)
                wandb.config = {
                "learning_rate": Learning_Rates_init[LR],
                "epochs": epochs,
                "batch_size": batch_size,
                "net_name": net_name,
                "ensemble_batch_size": ensemble_batch_size,
                "ENCORE_initial_epoch": SemiSup_initial_epoch,
                "Dataset": "Fold"+str(c),
                "affine": affine,

                }
                

                if 'Endovis' in project_name:
                    dataset_mode = 'folder'

                    dir_train_img = Dataset_Path_Train+str(Categories[c])+'/imgs'+'/Train'    
                    dir_train_mask = Dataset_Path_Train+str(Categories[c])+ mask_folder +'/Train'  
                    
                    dir_test_img = Dataset_Path_Test+str(Categories[c])+'/imgs'+ '/Test'
                    dir_test_mask = Dataset_Path_Test+str(Categories[c])+mask_folder+ '/Test'

                    if 'UnsupOnTest' in strategy:

                        dir_SemiTrain_img = Dataset_Path_Test+str(Categories[c])+'/imgs'+ '/Test'
                        dir_SemiTrain_mask = Dataset_Path_Test+str(Categories[c])+mask_folder+ '/Test'

                    elif 'UnsupOnTrain' in strategy:   

                        dir_SemiTrain_img = Dataset_Path_SemiTrain+str(Categories[c])+'/imgs'+'/Semi'
                        dir_SemiTrain_mask = Dataset_Path_SemiTrain+str(Categories[c])+mask_folder+'/Semi'

                elif 'Endometriosis' in project_name:

                    dataset_mode = 'folder'

                    dir_train_img = Dataset_Path_Train+str(Categories[c])+'/imgs'+'/Train'    
                    dir_train_mask = Dataset_Path_Train+str(Categories[c])+ mask_folder +'/Train'  
                    
                    dir_test_img = Dataset_Path_Test+str(Categories[c])+'/imgs'+ '/Test'
                    dir_test_mask = Dataset_Path_Test+str(Categories[c])+mask_folder+ '/Test'

                    dir_SemiTrain_img = Dataset_Path_SemiTrain+str(Categories[c])+'/imgs'+ '/Test'
                    dir_SemiTrain_mask = Dataset_Path_SemiTrain+str(Categories[c])+mask_folder+ '/Test' 
                    
                elif 'RETOUCH' in project_name or 'Cat3kToCaDIS' in project_name or 'MRI' in project_name:

                    dataset_mode = 'csv'


                    train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'.csv'    
                                       
                    test_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_test.csv'    

                    semi_train_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SemiSup.csv' 

                    SourceTest_IDs_CSV = Dataset_Path_Train+str(Categories[c])+'_SourceTest.csv' 

    


                # else:   #CATARACT

                #     dataset_mode = 'folder'   

                #     dir_train_img = Dataset_Path_Train+str(Categories[c])+'/imgs'  
                #     dir_train_mask = Dataset_Path_Train+str(Categories[c])+ mask_folder 
                    
                #     dir_test_img = Dataset_Path_Test+'/imgs'
                #     dir_test_mask = Dataset_Path_Test+mask_folder

                #     dir_SemiTrain_img = Dataset_Path_SemiTrain+'/imgs'
                #     dir_SemiTrain_mask = Dataset_Path_SemiTrain+mask_folder

                save_test = Results_path + Visualization_path + project_name + '_' + strategy +'_Thr_'+str(hard_label_thr)+'_'+net_name +"_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSup_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c])+'_'+'Affine_'+str(affine)+'/'
                dir_checkpoint = Results_path + Checkpoint_path + project_name + '_'+ strategy + '_'+net_name +'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c])+'/'
                csv_name = Results_path + CSV_path + project_name + '_' + strategy +'_Thr_'+str(hard_label_thr)+'_'+net_name +"_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSup_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size)+'_'+str(Categories[c])+'_'+'Affine_'+str(affine)+'.csv'
                create_directory(Results_path + Visualization_path)
                create_directory(Results_path + Checkpoint_path)
                create_directory(Results_path + CSV_path)


                net = Net1(n_classes=1, n_channels=3)
                logging.info(f'Network:\n'
                             f'\t{net.n_channels} input channels\n'
                             f'\t{net.n_classes} output channels (classes)\n')

                # if load:
                #     load_path_final = load_path+net_name +str(Learning_Rates_init[LR])+'_'+str(Categories[c])+'_'+'/CP_epoch'+str(load_epoch)+'.pth'
                    
                #     net.load_state_dict(
                #         torch.load(load_path_final, map_location=device)
                #     )
                #     logging.info(f'Model loaded from {load_path_final}')

                #     with open(CSV_path+net_name +str(Learning_Rates_init[LR])+'_'+str(Categories[c])+'_'+'.csv') as csv_file:
                #         csv_reader = csv.reader(csv_file, delimiter=',')
                        
                #         row_counter = 0
                #         for row in csv_reader:
                            
                #             if row_counter>0:
                #                 wandb.log({'Train_Loss': row[9],
                #                 'Test_Dice': row[0],
                #                 'Test_IoU': row[2]})
                #                 logging.info(f'wandb logged step {row_counter}')
                #             row_counter += 1
                #             if row_counter>load_epoch:
                #                 break
                             
                        
                            

                net.to(device=device)
                
                train_net(net=net,
                          epochs=epochs,
                          batch_size=batch_size,
                          lr=Learning_Rates_init[LR],
                          device=device)
            
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)