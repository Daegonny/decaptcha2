#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:56:02 2018

@author: daegonny
"""
import os
#import pandas as pd
import random
import torch
import numpy as np
from slide_window.net import Net
from PIL import Image
wd = os.getcwd()
os.chdir(wd)

#build window widths list
#widths = pd.read_csv("label/char_width.csv",sep=";")['width'].unique()
#widths = widths.tolist()
#widths.sort()

def img2tensor(img):
    img = img.convert('L')
    img = img.point(lambda p: p > 210 and 255)
    img = np.asarray(img)/255
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = img.type('torch.FloatTensor')
    img.unsqueeze_(0)
    return img

net = Net()
net = torch.load("slide_window/models/model.pt")
net.eval()

def predict_label(img, net):
    with torch.no_grad():
        outputs = net(img2tensor(img))
        _, predicted = torch.max(outputs, 1)
        result = int(predicted.data[0])
    return result

widths = [20,28,33]


random.seed(0)
chosen = []

for j in range(600):
    print("captcha: "+str(j))
    img = Image.open("img/captcha"+str(j)+".png")
    _, heigth = img.size
    idx = 1    
    first = 28
    last = first + 144
    position = first
    count = 0
    max_width = 32
    step = 1
    tryes = 0
    slices = []    
    factor = 1
    max_try = 5
    
    slice1 = False
    slice2 = False
    slice3 = False
    slice4 = False
    
    while len(slices) < 4:
        
        #corta primeira letra
        print("slice1")
        tryes = 0
        while not slice1:
            for width in widths:
                im1 = img.crop((position, 0, position + width, heigth))
                im1 = im1.resize((32,32))
                if(predict_label(im1, net)):
                    slices.append(im1)
                    position += int(width*factor)
                    slice1 = True
                    break
            tryes += 1 
            position += step
            if tryes > max_try and not slice1:
                slices.append(im1)
                position += int(width*factor)
                slice1 = True
                break
            
        #corta segunda letra
        print("slice2")
        tryes = 0
        while not slice2:
            for width in widths:
                im2 = img.crop((position, 0, position + width, heigth))
                im2 = im2.resize((32,32))
                if(predict_label(im2, net)):
                    slices.append(im2)
                    position += int(width*factor)
                    slice2 = True
                    break
            tryes += 1 
            position += step
            if tryes > max_try and not slice2:
                slices.append(im2)
                position += int(width*factor)
                slice2 = True
                break
            
        #corta terceira letra
        print("slice3")
        tryes = 0
        while not slice3:
            for width in widths:
                im3 = img.crop((position, 0, position + width, heigth))
                im3 = im3.resize((32,32))
                if(predict_label(im3, net)):
                    slices.append(im3)
                    position += int(width*factor)
                    slice3 = True
                    break
            position += step
            tryes += 1
            if tryes > max_try and not slice3:
                slices.append(im3)
                position += int(width*factor)
                slice3 = True
                break
            
        #corta quarta letra
        print("slice4")
        tryes = 0
        while not slice4:
            for width in widths:
                im4 = img.crop((position, 0, position + width, heigth))
                im4 = im4.resize((32,32))
                if(predict_label(im4, net)):
                    slices.append(im4)
                    position += int(width*factor)
                    slice4 = True
                    break
            position += step
            tryes += 1
            if tryes > max_try and not slice4:
                slices.append(im4)
                position += int(width*factor)
                slice4 = True
                break
       
    for sliced in slices:
        sliced.save("sliced/slice_"+str(j)+"_"+str(idx)+".png")
        idx += 1
    