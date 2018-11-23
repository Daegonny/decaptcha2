#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
import random

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
window_size= [11,18,19,21,23,24,25,26,27,28,29,30,31,32,32,32,33,34,34,36]
(winW, winH) = (window_size[0], 46)
crop = 0
i = 0

for resized in pyramid(image, scale=1.5):
	while crop!=4:
		for (x, y, window, final) in sliding_window(resized, stepSize=8, windowSize=(winW, winH)):
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			if (final):
				if(i < 19):
					i = i + 1
					print(window_size[i])
				(winW, winH) = (window_size[i], 46)
			# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
			# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
			# WINDOW

			rand = random.randint(0, 500)
			# print(crop)
			if(rand >=499):
				# cropImage()
				crop = crop +1
				cv2.rectangle(resized,(x, y), (x + winW, y + winH), (255, 255, 255),cv2.FILLED)

			cv2.imshow("windowSize", window)


			# since we do not have a classifier, we'll just draw the window
			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			time.sleep(0.0025)
			if(crop == 4):
				break
