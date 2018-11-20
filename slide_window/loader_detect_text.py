#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 11:27:46 2018

@author: daegonny
"""

import torch
#import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class DriveData(Dataset):
    __xs = []
    __ys = []
    threshold = 210

    def __init__(self, folder_dataset, transform=None, line_idxs=[0]):
        self.transform = transform
        # Open and load img path the whole training data
        count = 0
        self.__xs = []
        self.__ys = []
        with open('slide_window/label/text_non_text.csv') as f:
            for line in f:
                if count in line_idxs:
                    self.__xs.append("slide_window/text_non_text/slice"+str(count+1)+".png")
                    self.__ys.append(int(line))

                count = count + 1


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.__xs[index]).convert('L')
        img = img.point(lambda p: p > self.threshold and 255)
        if self.transform is not None:
            img = self.transform(img)


        # Convert image and label to torch tensors
        img = np.asarray(img)/255
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = img.type('torch.FloatTensor')
        label = torch.from_numpy(np.asarray(self.__ys[index]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)
