import SimpleITK as sitk
import numpy as np
import os
from PIL import Image

def makedir(folder):
    try:
        os.mkdir(folder)
    except:
        print(f'The folder {folder} exists.')  

def load_oct_seg(filename):
    """
    loads an .mhd file using simple_itk
    :param filename: 
    :return: 
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    ct_scan = ct_scan
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    print(spacing.shape)
    print(spacing)

    return ct_scan, origin, spacing

# img = sitk.ReadImage('RETOUCH-TrainingSet-Cirrus/TRAIN001/reference.mhd', imageIO="MetaImageIO")
# sitk.WriteImage(img, 'pngfile.png')
input_folder = 'RETOUCH-TrainingSet-Topcon/'
output_folder = 'img_dataset2/RETOUCH-TrainingSet-Topcon/'
volums = os.listdir(input_folder)
 
makedir(output_folder)
makedir(output_folder + 'imgs/')
makedir(output_folder + 'fluids/')
     

if 'Train' in input_folder:

    for i in range(len(volums)):
        imgs, _, _ = load_oct_seg(input_folder + volums[i] + '/oct.mhd')
        num_slices = imgs.shape[0]
        print(imgs.shape)
        # print(np.max(imgs))
        # print(np.min(imgs))
        for slice_num in range(0, num_slices):
            im_slice = imgs[slice_num, :, :]
            im_name = output_folder + 'imgs/' + str(volums[i]) + '_' + str(slice_num)+ '.png'
            # print(np.max(im_slice))
            # print(np.min(im_slice))
            if 'Spectralis' in input_folder: # range 0-2**16 instead of 0-255
                im_slice = im_slice.astype(np.float32)
                print(np.max(im_slice))
                im_slice = (im_slice / (2 ** 16) * 255)
                print(np.max(im_slice))
                #  
            im_slice = im_slice.astype(np.int8) 
            print(np.max(im_slice))  
            # sitk.WriteImage(im_slice, im_name)
            im = Image.fromarray(im_slice, mode='L')
            im.save(im_name)
            

        masks, _, _ = load_oct_seg(input_folder + volums[i] + '/reference.mhd')
        num_slices = masks.shape[0]

        for slice_num in range(0, num_slices):
            im_slice = masks[slice_num, :, :]
            mask_name = output_folder + 'fluids/' + str(volums[i]) + '_' + str(slice_num) + '.png'
            im = Image.fromarray(im_slice, mode='L')
            im.save(mask_name)  
else:

    for i in range(len(volums)):
        imgs, _, _ = load_oct_seg(input_folder + volums[i] + '/oct.mhd')
        num_slices = imgs.shape[0]

        for slice_num in range(0, num_slices):
            im_slice = imgs[slice_num, :, :]
            im_name = output_folder + 'imgs/' + str(volums[i]) + '_' + str(slice_num)+ '.png'
            # sitk.WriteImage(im_slice, im_name)
            im = Image.fromarray(im_slice, mode='L')
            # im.save(im_name)






# img, _, _ = load_oct_seg('RETOUCH-TrainingSet-Cirrus/TRAIN001/oct.mhd')
# num_slices = img.shape[0]
# for slice_num in range(0, num_slices):
#     im_slice = img[slice_num, :, :]
    
# print(num_slices)