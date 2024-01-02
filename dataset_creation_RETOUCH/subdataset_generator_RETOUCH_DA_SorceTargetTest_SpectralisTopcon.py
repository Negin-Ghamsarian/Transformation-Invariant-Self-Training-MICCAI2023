import os
import csv
import glob
import random
import pandas as pd

def ID_shuffler(list1, list2):

        temp = list(zip(list1, list2))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        # res1 and res2 come out as tuples, and so must be converted to lists.
        res1, res2 = list(res1), list(res2)

        return res1, res2

def csv_saver(save_path, a, b, c, d, e, f, g, h, i, name='SpectralisVsTopcon4'):

        

        dict = {"imgs": a , "masks": b }
        df = pd.DataFrame(dict)
        df.to_csv(save_path + name + '_' + str(i)+'.csv')

        dict = {"imgs": c , "masks": d}
        df = pd.DataFrame(dict)
        df.to_csv(save_path + name + '_' + str(i)+ '_SemiSup' + '.csv')

        dict = {"imgs": e , "masks": f}
        df = pd.DataFrame(dict)
        df.to_csv(save_path + name + '_' + str(i)+ '_test' + '.csv')     

        dict = {"imgs": g , "masks": h}
        df = pd.DataFrame(dict)
        df.to_csv(save_path + name + '_' + str(i)+ '_SourceTest' + '.csv')     

save_path = '/storage/homefs/ng22l920/Codes/Semi_Supervised_ENCORE/Semi_Supervised_ENCORE_MICCAI23/TrainIDs_RETOUCH_DA/'

Topcon_dir = '/storage/workspaces/artorg_aimi/ws_00000/Negin/RETOUCH/img_dataset2/RETOUCH-TrainingSet-Topcon/'
cases_Topcon = [50, 51, 52, 53, 54, 56, 59, 60, 61, 62, 64, 65, 66, 67, 68, 70]
folds_Topcon = [[50, 51, 52], [54, 56, 61], [62, 66, 67], [60, 68, 70]]


Spectralis_dir = '/storage/workspaces/artorg_aimi/ws_00000/Negin/RETOUCH/img_dataset2/RETOUCH-TrainingSet-Spectralis/'
cases_Spectralis = [25, 26, 27, 29, 30, 32, 33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
# folds_Spectralis = [[46, 47, 48, 26], [25, 27, 29, 30], [32, 33, 36, 43], [34, 39, 44, 45]]
# folds_Spectralis = [[46, 47, 48, 26, 25, 27, 29, 30], [25, 27, 29, 30, 32, 33, 36, 43], [32, 33, 36, 43, 34, 39, 26, 45], [43, 39, 44, 45, 46, 47, 48, 26]]
# folds_Spectralis = [[25, 27, 29, 30, 32, 33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45], [26, 32, 33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], [25, 26, 27, 29, 30, 34, 37, 39, 40, 41, 42, 44, 45, 46, 47, 48], [25, 26, 27, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 47, 48]]
folds_Spectralis = [[25, 26, 27, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 44, 47, 48], [25, 26, 27, 29, 30, 34, 37, 39, 40, 41, 42, 44, 45, 46, 47, 48],
[26, 32, 33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], [25, 27, 29, 30, 32, 33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45]]

image_dir = 'imgs_IRF/'
mask_dir = 'IRF/'

for i in range(len(folds_Topcon)):
    locals()['imgs_Topcon_test_fold_' + str(i)] = []
    locals()['masks_Topcon_test_fold_' + str(i)] = []

    locals()['imgs_Spectralis_train_fold_' + str(i)] = []
    locals()['masks_Spectralis_train_fold_' + str(i)] = []

    locals()['imgs_Spectralis_test_fold_' + str(i)] = []
    locals()['masks_Spectralis_test_fold_' + str(i)] = []

    locals()['imgs_Topcon_semi_fold_' + str(i)] = []
    locals()['masks_Topcon_semi_fold_' + str(i)] = []

    fold_topcon = folds_Topcon[i]
    fold_spectralis = folds_Spectralis[i]
    fold_semi_topcon = cases_Topcon.copy()
    fold_test_spectralis = cases_Spectralis.copy()

    for k in range(len(fold_topcon)):
        fold_semi_topcon.remove(fold_topcon[k])

    for k in range(len(fold_spectralis)):  
        print(f'cases_Spectralis: {cases_Spectralis}')
        print(f'fold_test_spectralis: {fold_test_spectralis}')
        print(f'fold_spectralis[k]: {fold_spectralis[k]}')
        fold_test_spectralis.remove(fold_spectralis[k])  


    for j in range(len(fold_topcon)):
        for f in glob.glob(Topcon_dir + image_dir + 'TRAIN0' + str(fold_topcon[j]) + '*.png'):
            locals()['imgs_Topcon_test_fold_' + str(i)].append(f)

        for f in glob.glob(Topcon_dir + mask_dir + 'TRAIN0' + str(fold_topcon[j]) + '*.png'):
            locals()['masks_Topcon_test_fold_' + str(i)].append(f)   

        locals()['imgs_Topcon_test_fold_' + str(i)].sort()
        locals()['masks_Topcon_test_fold_' + str(i)].sort()    

        locals()['imgs_Topcon_test_fold_' + str(i)], locals()['masks_Topcon_test_fold_' + str(i)] \
            = ID_shuffler(locals()['imgs_Topcon_test_fold_' + str(i)], locals()['masks_Topcon_test_fold_' + str(i)]) 



    for j in range(len(fold_spectralis)):    
        for f in glob.glob(Spectralis_dir + image_dir + 'TRAIN0' + str(fold_spectralis[j]) + '*.png'):
            locals()['imgs_Spectralis_train_fold_' + str(i)].append(f)    

        for f in glob.glob(Spectralis_dir + mask_dir + 'TRAIN0' + str(fold_spectralis[j]) + '*.png'):
            locals()['masks_Spectralis_train_fold_' + str(i)].append(f)     

        locals()['imgs_Spectralis_train_fold_' + str(i)].sort()  
        locals()['masks_Spectralis_train_fold_' + str(i)].sort()

        locals()['imgs_Spectralis_train_fold_' + str(i)], locals()['masks_Spectralis_train_fold_' + str(i)] \
            = ID_shuffler(locals()['imgs_Spectralis_train_fold_' + str(i)], locals()['masks_Spectralis_train_fold_' + str(i)])



    for j in range(len(fold_semi_topcon)): 

        for f in glob.glob(Topcon_dir + image_dir + 'TRAIN0' + str(fold_semi_topcon[j]) + '*.png'):
            locals()['imgs_Topcon_semi_fold_' + str(i)].append(f)

        for f in glob.glob(Topcon_dir + mask_dir + 'TRAIN0' + str(fold_semi_topcon[j]) + '*.png'):
            locals()['masks_Topcon_semi_fold_' + str(i)].append(f)   

        locals()['imgs_Topcon_semi_fold_' + str(i)].sort()
        locals()['masks_Topcon_semi_fold_' + str(i)].sort()    

        locals()['imgs_Topcon_semi_fold_' + str(i)], locals()['masks_Topcon_semi_fold_' + str(i)] \
            = ID_shuffler(locals()['imgs_Topcon_semi_fold_' + str(i)], locals()['masks_Topcon_semi_fold_' + str(i)]) 


    for j in range(len(fold_test_spectralis)): 

        for f in glob.glob(Spectralis_dir + image_dir + 'TRAIN0' + str(fold_test_spectralis[j]) + '*.png'):
            locals()['imgs_Spectralis_test_fold_' + str(i)].append(f)

        for f in glob.glob(Spectralis_dir + mask_dir + 'TRAIN0' + str(fold_test_spectralis[j]) + '*.png'):
            locals()['masks_Spectralis_test_fold_' + str(i)].append(f)   

        locals()['imgs_Spectralis_test_fold_' + str(i)].sort()
        locals()['masks_Spectralis_test_fold_' + str(i)].sort()    

        locals()['imgs_Spectralis_test_fold_' + str(i)], locals()['masks_Spectralis_test_fold_' + str(i)] \
            = ID_shuffler(locals()['imgs_Spectralis_test_fold_' + str(i)], locals()['masks_Spectralis_test_fold_' + str(i)])         


    csv_saver(save_path, locals()['imgs_Spectralis_train_fold_' + str(i)], locals()['masks_Spectralis_train_fold_' + str(i)],\
        locals()['imgs_Topcon_semi_fold_' + str(i)], locals()['masks_Topcon_semi_fold_' + str(i)],\
        locals()['imgs_Topcon_test_fold_' + str(i)], locals()['masks_Topcon_test_fold_' + str(i)],\
            locals()['imgs_Spectralis_test_fold_' + str(i)], locals()['masks_Spectralis_test_fold_' + str(i)], i)        







    
    
    
    # print(f'____________________________________imgs_Topcon_fold_{i}_____________________________________________')   
    # print(locals()['imgs_Topcon_fold_' + str(i)])     

    # print(f'____________________________________masks_Topcon_fold_{i}_____________________________________________')   
    # print(locals()['masks_Topcon_fold_' + str(i)])  

   

















