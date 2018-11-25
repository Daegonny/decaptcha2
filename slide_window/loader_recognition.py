#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 09:23:46 2018

@author: daegonny
"""

import torch
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
        with open('label/label.csv') as f:
            for line in f:
                if count in line_idxs:
                    self.__xs.append("sliced/slice_"+str(count)+"_1.png")
                    self.__xs.append("sliced/slice_"+str(count)+"_2.png")
                    self.__xs.append("sliced/slice_"+str(count)+"_3.png")
                    self.__xs.append("sliced/slice_"+str(count)+"_4.png")

                    self.__ys.append(self.translate_label(line[0]))
                    self.__ys.append(self.translate_label(line[1]))
                    self.__ys.append(self.translate_label(line[2]))
                    self.__ys.append(self.translate_label(line[3]))

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

    def translate_label(self,label):
        label = label.upper()
        if label == "1":
            return 0

        if label == "2":
            return 1

        if label == "3":
            return 2

        if label == "4":
            return 3

        if label == "5":
            return 4

        if label == "6":
            return 5

        if label == "7":
            return 6

        if label == "8":
            return 7

        if label == "9":
            return 8

        if label == "B":
            return 9

        if label == "C":
            return 10

        if label == "D":
            return 11

        if label == "F":
            return 12

        if label == "G":
            return 13

        if label == "H":
            return 14

        if label == "J":
            return 15

        if label == "K":
            return 16

        if label == "L":
            return 17

        if label == "M":
            return 18

        if label == "N":
            return 19

        if label == "P":
            return 20

        if label == "R":
            return 21

        if label == "S":
            return 22

        if label == "T":
            return 23

        if label == "V":
            return 24

        if label == "W":
            return 25

        if label == "X":
            return 26


        if label == "Z":
            return 27
