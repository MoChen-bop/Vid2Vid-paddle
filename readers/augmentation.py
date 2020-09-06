import os
import sys
sys.path.append('/home/aistudio')

import random
from PIL import Image
import numpy as np 


class Compose():

    def __init__(self, transforms):
        self.transforms = transforms
    

    def __call__(self, image, params):
        for t in self.transforms:
            image = t(image, params)
        return image


class ScaleImage():
    def __init__(self, method=Image.BICUBIC):
        self.method = method
    

    def __call__(self, image, params):
        w, h = params['new_size']
        return image.resize((w, h), self.method)


class Crop():
    def __init__(self, ):
        pass
    

    def __call__(self, image, params):
        ow, oh = image.size
        x1, y1 = params['crop_pos']
        tw, th = params['crop_size']
        return image.crop((x1, y1, x1 + tw, y1 + th))


class Flip():
    def __init__(self, ):
        pass
    

    def __call__(self, image, params):
        flip = params['flip']
        if flip:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return image


class ColorAUG():
    def __init__(self, ):
        pass
    

    def __call__(self, image, params):
        colors = params['color_aug']
        h, s, v = image.convert('HSV').split()
        h = h.point(lambda i: (i + colors[0]) % 256)
        s = s.point(lambda i: min(255, max(0, i * colors[1] + colors[2])))
        v = v.point(lambda i: min(255, max(0, i * colors[3] + colors[4])))
        image = Image.merge('HSV', (h, s, v)).convert('RGB')

        return image
    

class Normalize():
    def __init__(self, ):
        pass

    
    def __call__(self, image, params):
        image = np.array(image).astype('float32')
        image /= 255.0
        image -= params['mean']
        image /= params['std']
        image = image.transpose((2, 0, 1))
        return image


class ToTensor():

    def __init__(self, ):
        pass
    

    def __call__(self, image, params):
        image = np.array(image).astype('float32')
        if len(image.shape) < 3:
            return image[np.newaxis,:]
        else:
            return image


class BaseTransform():
    def __init__(self, method=Image.NEAREST, is_label=False):
        if is_label:
            self.transforms = Compose([
                ScaleImage(method),
                ToTensor()
            ])
        else:
            self.transforms = Compose([
                ScaleImage(method),
                Normalize()
            ])


    def __call__(self, image, params):
        return self.transforms(image, params)


class Augmentation():
    def __init__(self, method=Image.NEAREST, is_label=False):
        if is_label:
            self.transforms = Compose([
                ScaleImage(method),
                Crop(),
                Flip(),
                ToTensor()
            ])
        else:
            self.transforms = Compose([
                ScaleImage(method),
                Crop(),
                Flip(),
                ColorAUG(),
                Normalize()
            ])


    def __call__(self, image, params):
        return self.transforms(image, params)
