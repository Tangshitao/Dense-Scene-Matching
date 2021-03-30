import torch
import cv2
import torch.nn as nn
from PIL import Image
import PIL
import numpy as np
import random
from .geometry import scale_K
import math


class Resize(object):
    def __init__(self, size):
        if type(size) is not tuple:
            self.h = size
            self.w = size
        else:
            self.h, self.w = size

    def __call__(self, img, depth, Tcw, K):
        h, w = depth.shape

        if h <= w:
            scale = 1.0 * self.h / h
        else:
            scale = 1.0 * self.h / w
        K = scale_K(K, scale)

        if img is not None:
            img = Image.fromarray(img)
            if h <= w:
                img = img.resize((self.w, self.h), resample=Image.ANTIALIAS)
            else:
                img = img.resize((self.h, self.w), resample=Image.ANTIALIAS)
            img = np.asarray(img)

        depth = Image.fromarray(depth)
        if h <= w:
            depth = depth.resize((self.w, self.h), resample=Image.NEAREST)
        else:
            depth = depth.resize((self.h, self.w), resample=Image.NEAREST)
        depth = np.asarray(depth)

        return img, depth, Tcw, K

    def random_parameters(self):
        pass


class RandomCrop(object):
    def __init__(self, min_ratio=0.8, max_ratio=1):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img, depth, Tcw, K):
        if img is not None:
            h, w, _ = img.shape
            scale_ratio = (
                random.random() * (self.max_ratio - self.min_ratio) + self.min_ratio
            )
            left_up_ratio = random.random() * (1 - scale_ratio)
            x = int(w * left_up_ratio)
            y = int(h * left_up_ratio)

            new_h = int(h * scale_ratio)
            new_w = int(w * scale_ratio)

            img = img[y : y + new_h, x : x + new_w, :]
            depth = depth[y : y + new_h, x : x + new_w]

            K[0, 2] = K[0, 2] - x
            K[1, 2] = K[1, 2] - y

        return img, depth, Tcw, K

    def random_parameters(self):
        pass


class RandomCenterCrop(object):
    def __init__(self, min_ratio=0.8, max_ratio=1):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img, depth, Tcw, K):
        if img is not None:
            h, w, _ = img.shape

            g = math.gcd(h, w)
            unit_h = h / g
            unit_w = w / g

            x = int(int(w * (1 - self.scale_ratio) / 2 / unit_w) * unit_w)
            y = int(int(h * (1 - self.scale_ratio) / 2 / unit_h) * unit_h)

            img = img[y : h - y, x : w - x, :]
            depth = depth[y : h - y, x : w - x]

            K[0, 2] = (w - 2 * x) / 2
            K[1, 2] = (h - 2 * y) / 2

        return img, depth, Tcw, K

    def random_parameters(self):
        self.scale_ratio = (
            random.random() * (self.max_ratio - self.min_ratio) + self.min_ratio
        )


class CenterCrop(object):
    def __init__(self, scale_ratio=0.9):
        self.scale_ratio = scale_ratio

    def __call__(self, img, depth, Tcw, K):
        if img is not None:
            h, w, _ = img.shape

            x = int(w * (1 - self.scale_ratio) / 2)
            y = int(h * (1 - self.scale_ratio) / 2)

            new_h = int(h * self.scale_ratio)
            new_w = int(w * self.scale_ratio)
            # print(x,y,new_h,new_w)

            img = img[y : y + new_h, x : x + new_w, :]
            depth = depth[y : y + new_h, x : x + new_w]

            K[0, 2] = new_w / 2
            K[1, 2] = new_h / 2

        return img, depth, Tcw, K

    def random_parameters(self):
        pass


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor(
                [
                    [0.4009, 0.7192, -0.5675],
                    [-0.8140, -0.0045, -0.5808],
                    [0.4203, -0.6948, -0.5836],
                ]
            )
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor, depth, Tcw, K):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros(*self.eig_val.size())) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)

        tensor = tensor + quatity.view(3, 1, 1)
        return tensor, depth, Tcw, K


class Normalize(object):
    def __init__(self, scale=1, mean=[0, 0, 0], std=[1, 1, 1]):
        # super(Resize).__init__()
        self.scale = scale
        self.mean = np.array(mean).reshape(1, 3)
        self.std = np.array(std).reshape(1, 3)

    def __call__(self, img, depth, Tcw, K):
        if img is not None:
            ori_shape = img.shape
            img = img.reshape(-1, 3)

            img = (img / self.scale - self.mean) / self.std
            img = img.reshape(ori_shape)
        return img, depth, Tcw, K

    def random_parameters(self):
        pass


class ToTensor(object):
    def __init__(self):
        # super(ToTensor).__init__()
        pass

    def __call__(self, img, depth, Tcw, K):

        return (
            torch.from_numpy(img).permute(2, 0, 1),
            torch.from_numpy(depth),
            torch.from_numpy(Tcw),
            torch.from_numpy(K),
        )

    def random_parameters(self):
        pass


class Compose(object):
    def __init__(self, transforms):
        # super(Compose).__init__()
        self.transforms = transforms

    def __call__(self, img, depth, Tcw, K):
        for tsf in self.transforms:
            img, depth, Tcw, K = tsf(img, depth, Tcw, K)
        return img, depth, Tcw, K

    def random_parameters(self):
        for tsf in self.transforms:
            tsf.random_parameters()
