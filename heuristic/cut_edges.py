import os
from PIL import Image
import numpy as np
import pandas as pd

os.chdir("D:/Users/igor.marques/Documents/chagasm/teste")

count=2
threshold = 215
step = 6

im = Image.open("img/captcha"+str(count)+".png").convert('L')
im = im.point(lambda p: p > threshold and 255)

im.save("img/captcha"+str(count)+"p.png")

im.getpixel((0,0))

width, height = im.size


columns = []
for x in range(width):
    whites = 0
    is_letter = False
    for y in range(height):
        if im.getpixel((x,y)) == 255:
            whites += 1
            if y >= 6 and y <= 41:
                is_letter = True

    columns.append((whites,is_letter))


first = 0
last = 0

#find first cut
for idx in range(width):
    if columns[idx+2][0] > 2 and columns[idx+2][1]:
        first = idx
        break

#find last cut
for idx in range(width):
    if columns[width-(idx+3)][0] > 2 and columns[width-(idx+3)][1]:
        last = width-(idx+1)
        break
