#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:56:02 2018

@author: daegonny
"""
import os
import pandas as pd
import random
from PIL import Image
wd = os.getcwd()
os.chdir(wd)

#build window widths list
widths = pd.read_csv("label/char_width.csv",sep=";")['width'].unique()
widths = widths.tolist()
widths.sort()

idx = 1
random.seed(0)
chosen = []

for j in range(600):

    img = Image.open("img/captcha"+str(j)+".png")
    _, heigth = img.size
    
    first = 28
    last = first + 144
    position = first
    count = 0
    max_width = 32    
    
    for i in range(20):
        
        rd = random.randint(0,16)
        chosen.append(rd)
        width = widths[rd]
        im = img.crop((position, 0, position + width, heigth))
        im = im.resize((32,32))
        im.save("slide_window/text_non_text/slice"+str(idx)+".png")
        
        idx = idx + 1
        count = count + 1
        position = position + (count*2)
        if (position + max_width/2) > last: #out of max width
            count = 0
            position = first + (count*2)