# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from slide_window.net import Net
from PIL import Image
wd = os.getcwd()
os.chdir(wd)

classes = ('1', '2', '3', '4','5', '6', '7', '8', '9', 'B','C','D',
               'F','G','H','J','K','L','M','N','P','R','S','T','V',
               'W','X','Z')

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

net2 = Net()
net2 = torch.load("slide_window/models/model_recognition2.pt")
net2.eval()

def predict_label(img, net):
    with torch.no_grad():
        outputs = net(img2tensor(img))
        _, predicted = torch.max(outputs, 1)
        result = int(predicted.data[0])
    return result

def predict_label2(img, net):
    with torch.no_grad():
        outputs = net(img2tensor(img))
        _, predicted = torch.max(outputs, 1)
        result = int(predicted.data[0])
    return result

def get_text(img, net):
    widths = [20,28,33]
    _, heigth = img.size   
    first = 28
    position = first
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
        #print("slice1")
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
        #print("slice2")
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
        #print("slice3")
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
        #print("slice4")
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
    
    text = ""
    for sliced in slices:
        text += classes[predict_label2(sliced, net2)]
    return text

hits = 0
total = 0
with open('label/label.csv') as f:
    for line in f:
        if total < 600:
            img = Image.open("img/captcha"+str(total)+".png")
            #img.show()
            text = get_text(img, net)
            total += 1
            #print(line, text)
            if text == line[0:4]:
                hits += 1

print(hits/total)                