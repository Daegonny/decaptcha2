#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from slide_window_func.pyimagesearch.helpers import pyramid
from slide_window_func.pyimagesearch.helpers import sliding_window
import argparse
import numpy as np
import time
import cv2
import torch
from slide_window.slide_window import predict_label
from slide_window.net import Net

def pos(vet):
	j=1
	vet.sort(key=lambda y: y[1])
	for i in vet:
		cv2.imwrite("cropped_slide_window/cropped"+str(j)+".png", i[0])
		j = j +1

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
#args = vars(ap.parse_args())

image = cv2.imread("slide_window_func/images/captcha98.png")
window_size= [20,28,33,20,20,28,33,20,20,28,33,20,20,28,33,20,20,28,33,20]
(winW, winH) = (window_size[0], 46)
crop = 0
i = 0
vet = list()
net = Net()
net = torch.load("slide_window/models/model.pt")
net.eval()

for resized in pyramid(image, scale=1.5):
	while crop!=4:
		for (x, y, window, final) in sliding_window(resized, stepSize=28, windowSize=(winW, winH)):
			if window.shape[0] != winH or window.shape[1] != winW:
				(x, y, window, final) -
			if (final):
				i = i + 1
				print(i)
				if(i < 18):
					print(window_size[i])
				if(i > 18):
					print("ahah")
					i = 0
				(winW, winH) = (window_size[i], 46)
			clone = window.copy()
			clone = cv2.resize(clone, (32, 32))
			rand = predict_label(clone, net)
			# print(crop)
			if(rand ==1):
				vet.append([clone,x])
				cv2.imshow("teste", clone)
				crop = crop + 1
				print(winW)
				cv2.rectangle(resized,(x, y), (x + winW+5, y + winH), (255/360 * 30, 255/360 * 30, 255/360 * 30),cv2.FILLED)
			

			cv2.imshow("windowSize", window)


			# since we do not have a classifier, we'll just draw the window
			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			time.sleep(0.05)
			if(crop == 4):
				pos(vet)
				break
