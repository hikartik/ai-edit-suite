import os
import glob
import torch
import random

import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask
import cv2
from .segmentor_fcn import segmentor, fill_gaps


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS
        self.device = config.SEG_DEVICE
        self.objects = config.OBJECTS
        self.segment_net = config.SEG_NETWORK
        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except Exception as e:
            print(f'loading error: {self.data[index]}, error: {e}')
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)
        
    def load_size(self, index):
        img = Image.open(self.data[index])
        width, height = img.size
        return width, height


    def load_item(self, index):

        size = self.input_size

        # load image
        img = Image.open(self.data[index])
        
        # gray to rgb
        if img.mode != 'RGB':
            img = gray2rgb(np.array(img))
            img = Image.fromarray(img)

        # segment and get mask
        img_arr, mask_arr = segmentor(self.segment_net, img, self.device, self.objects)

        # normalize size to a (w, h) tuple
        if isinstance(size, int):
            size_tuple = (size, size)
        else:
            size_tuple = tuple(size)

        # resize image
        img_pil = Image.fromarray(img_arr)
        img_resized = img_pil.resize(size_tuple, getattr(Image, "Resampling", Image).LANCZOS)
        img = np.array(img_resized)

        # create grayscale image
        img_gray = rgb2gray(img)

        # resize mask
        mask_pil = Image.fromarray(mask_arr)
        mask_resized = mask_pil.resize(size_tuple, getattr(Image, "Resampling", Image).LANCZOS)
        mask = np.array(mask_resized)

        # binarize mask to 0 or 255
        idx = (mask > 0)
        mask[idx] = 255

        # apply gap filling
        mask = np.apply_along_axis(fill_gaps, 1, mask)  # horizontal padding
        mask = np.apply_along_axis(fill_gaps, 0, mask)  # vertical padding

        # load edge
        edge = self.load_edge(img_gray, index, mask)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask_param = None if self.training else (1 - mask / 255).astype(bool)
        
        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask_param).astype(float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resized(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask_param)

            return edge

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
