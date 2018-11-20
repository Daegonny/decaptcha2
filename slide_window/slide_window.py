#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Nov 20 12:15:04 2018

@author: daegonny
"""
import torch
import numpy as np
from slide_window.net import Net
from PIL import Image
import os

wd = os.getcwd()
os.chdir(wd)

def img2tensor(img_path):
    img = Image.open(img_path).convert('L')
    img = img.point(lambda p: p > 210 and 255)

    # Convert image and label to torch tensors
    img = np.asarray(img)/255
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = img.type('torch.FloatTensor')
    img.unsqueeze_(0)
    return img

net = Net()
net = torch.load("slide_window/models/model.pt")
net.eval()


for i in range(499):
    
    img_path = "slide_window/text_non_text/slice"+str(i+1)+".png"
    inputs = img2tensor(img_path)
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
    print(int(predicted.data[0]))