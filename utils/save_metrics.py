# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:24:39 2020

@author: Negin
"""
import csv 

def save_metrics(values, name):
    
    fields = ['ave(dice)', 'std(dice)', 'ave(IoU)', 'std(IoU)', 'min(dice)', 'min(IoU)', 'max(dice)', 'max(IoU)', 'inference_time', 'epoch_loss']  
    
    with open(name, 'w') as f: 
          
        # using csv.writer method from CSV package 
        write = csv.writer(f)       
        write.writerow(fields) 
        write.writerows(values) 
        
        
   
    
