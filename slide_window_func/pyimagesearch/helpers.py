#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imutils

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(28, image.shape[1], stepSize):
			# yield the current window
			if(x >= image.shape[1]-windowSize[1]):
				yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],True)
			else:
				yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],False)