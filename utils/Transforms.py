#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:24:58 2022

@author: negin
"""

# =============================================================================
# Tensor-enabled image-mask transformations with PyTorch 
 # List of transformers:
 #    1) Compose
 #    2) RandomApply
 #    3) RandomResize
 #    4) Resize
 #    5) RandomGrayscale
 #    6) RandomHorizontalFlip
 #    7) RandomVerticalFlip
 #    8) RandomCrop
 #    9) CenterCrop
 #    10) ToTensorv
 #    11) Normalize
 #    12) RemoveWhitelines
 #    13) RandomRotation
 #    14) GaussianBlur
 #    15) ColorJitter
 #    16) GaussianNoise
 #    17) RandomResizedCrop
 #    18) RandomAdjustSharpness      
# =============================================================================
import torch
import torchvision
#from torchvision import transforms as T
from torchvision.transforms import functional as F
import numbers
import math
import warnings
import random
import numpy as np


torch.manual_seed(100)
random.seed(100)
np.random.seed(100)


def pad_if_smaller(img, mask, size, fill=0):
    ow = img.shape[1]
    oh = img.shape[2]
    min_size = min(oh, ow)
    if min_size < size:
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        # print(f'padh: {padh}')
        # print(f'padw: {padw}')
        img = F.pad(img, [int(padw/2), int(padh/2), int(padw-padw/2), int(padh-padh/2)], fill=fill)
        mask = F.pad(mask, [int(padw/2), int(padh/2), int(padw-padw/2), int(padh-padh/2)], fill=fill)
    return img, mask



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target



class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.p = p
        self.transforms = transforms

    def __call__(self, img, mask):
        if self.p < torch.rand(1)[0]:
            return img, mask
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomApply_Customized(object):
    def __init__(self, transforms, p):
        super().__init__()
        self.p = p
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            if self.p > torch.rand(1)[0]:
               img, mask = t(img, mask)
        return img, mask

class One_Of(object):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.l = len(transforms)
    def __call__(self, img, mask):
        selected = torch.randint(self.l, (1,))[0]
        img, mask = self.transforms[selected](img, mask)
        return img, mask


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = torch.randint(self.min_size, self.max_size, (1,))[0]
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return image, target

class Resize_KeepRatio(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return image, target        



class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, target):
        num_output_channels = F.get_image_num_channels(image)
        if torch.rand(1)[0] < self.p:
            image = F.rgb_to_grayscale(image, num_output_channels=num_output_channels)
        return image, target



class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if torch.rand(1)[0] < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if torch.rand(1)[0] < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, self.size[0] - h)
        pad_lr = max(0, self.size[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = torch.randint(0, h - self.size[0], (1,))[0]
        j = torch.randint(0, w - self.size[1], (1,))[0]
        image = image[:, i:i + self.size[0], j:j + self.size[1]]
        mask = mask[:, i:i + self.size[0], j:j + self.size[1]]
        return image, mask


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target.astype(float)).long()
        # target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RemoveWhitelines(object):
    def __call__(self, image, target):
        target = torch.where(target == 255, 0, target)
        return image, target


class RandomRotation(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = torch.rand(1)[0] * 2 * self.degree - self.degree
        rotate_degree = float(rotate_degree)
        return F.rotate(img, rotate_degree), F.rotate(mask, rotate_degree)


class GaussianBlur(object):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img, mask):
        return F.gaussian_blur(img, self.kernel_size, self.sigma), mask

# My RandomAdjustSharpness is different from PyTorch. My function randomly selects a float number between 1 and the given number as the sharpness factor

class RandomAdjustSharpness(object):
    def __init__(self, sharpness_factor):
        self.sharpness_factor = sharpness_factor
        self.r1 = 1
        self.r2 = sharpness_factor       

    def __call__(self, img, mask):

        sharpnessFactor = (self.r1 - self.r2) * torch.rand(1)[0] + self.r2
        return F.adjust_sharpness(img, sharpnessFactor), mask
        
class ColorJitter(object):
    """Mostly from the docs"""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, img, mask):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img, mask


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, img, mask):
        return img + torch.randn(img.size()) * self.std + self.mean, mask

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomResizedCrop(object):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
        super().__init__()
        
        self.size = size
        

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
    
    def get_params(self, img):

        width, height = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        mask = F.resized_crop(mask, i, j, h, w, self.size, self.interpolation)
        return img, mask


    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


if __name__ == '__main__':
    a = torch.rand((1, 3, 512, 512))
    # RandomRotate(20)(a, a)
    t = Compose([RandomApply([RandomRotation(30), GaussianNoise()]), ColorJitter(brightness=1)])
    t(a, a[:,0])