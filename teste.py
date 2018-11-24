# -*- coding: utf-8 -*-

import cv2
from PIL import Image
import numpy as np

img = Image.fromarray(clone)
img = Image.fromarray(cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB))
#img2 = Image.fromarray(np.asarray(img1))

img.show("haha")