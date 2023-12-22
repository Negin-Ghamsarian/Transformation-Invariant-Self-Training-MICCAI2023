import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
import shutil

def makedir(folder):
    try:
        os.mkdir(folder)
    except:
        print(f'The folder {folder} exists.')  

scanner_dir = '/storage/workspaces/artorg_aimi/ws_00000/Negin/RETOUCH/img_dataset2/RETOUCH-TrainingSet-Cirrus/'
mask_folder = 'fluids/'
# img_folder = 'imgs/'
img_folder = 'imgs_IRF/'
makedir(scanner_dir + img_folder)
makedir(scanner_dir + 'IRF/')

imlist = os.listdir(scanner_dir+mask_folder)

IRF_CODE = 1
SRF_CODE = 2
PED_CODE = 3


for i in range(len(imlist)):

    mask = (plt.imread(scanner_dir+mask_folder+imlist[i])*255).astype(np.uint8)
    print(np.unique(mask==IRF_CODE))
    print(len(np.unique(mask==IRF_CODE)))

    if len(np.unique(mask==IRF_CODE)) == 1:
        print(f'imlist[i]: {imlist[i]}')

        # os.remove(scanner_dir+mask_folder+imlist[i])
        # os.remove(scanner_dir+img_folder+imlist[i])    
    else:    
        # shutil.move(scanner_dir+img_folder+imlist[i], scanner_dir+img_folder_fluid+imlist[i])

        IRF = (mask == IRF_CODE).astype(np.uint8)
        # PED = (mask == PED_CODE).astype(np.uint8)
        # summed = (mask>0).astype(np.uint8)

        # print(np.unique((IRF*255).astype(np.uint8)))
        # print(np.unique((SRF*255).astype(np.uint8)))
        # print(np.unique((PED*255).astype(np.uint8)))

        plt.imsave(scanner_dir+'IRF/'+imlist[i], IRF, cmap=cm.gray)
        # plt.imsave(scanner_dir+'PED/'+imlist[i], PED, cmap=cm.gray)

        shutil.copy(scanner_dir+'imgs/'+imlist[i], scanner_dir+img_folder+imlist[i])






