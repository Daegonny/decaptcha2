#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from slide_window_func.pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
import torch
from slide_window.slide_window import predict_label
from slide_window.net import Net

def pos(vet, cp):
	j=1
	vet.sort(key=lambda y: y[1])
	for i in vet:
		cv2.imwrite("cropped_slide_window/cropped_"+str(cp)+"_"+str(j)+".png", i[0])
		j = j +1

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
#args = vars(ap.parse_args())


window_size= [20,28,33]
#window_size= [11,18,19,21,23,24,25,26,27,28,29,30,31,32,32,32,33,34,34,36]
(winW, winH) = (20, 46)
crop = 0
i = 0
vet = list()
net = Net()
net = torch.load("slide_window/models/model.pt")
net.eval()

qtd_captchas = 0
fileName = []
with open('label/label.csv') as f:
	for line in f:
		fileName.append(str(line))
		qtd_captchas = qtd_captchas + 1

for cp in range(1):
	image = cv2.imread("img/captcha"+str(cp)+".png")
	crop = 0
	i=0
	vet.clear()
	step = 32
	(winW, winH) = (window_size[i], 46)
	while crop<=4:
		for (x, y, window, final) in sliding_window(image, stepSize=step, windowSize=(winW, winH)):
			if window.shape[0] != winH or window.shape[1] != winW:
				if (final):
					i = i + 1
					if(i > 2):
						i = 0
						step = step - 1
						print("sahsahu")
				continue
			if (final):
				i = i + 1
				if(i > 2):
					i = 0
					step = step - 1
					print("sahsahu 2")
				(winW, winH) = (window_size[i], 46)
			clone = window.copy()
			clone = cv2.resize(clone, (32, 32))
			rand = predict_label(clone, net)
			# print(crop)
			if(rand ==1):
				vet.append([clone,x])
				cv2.imshow("teste", clone)
				crop = crop + 1
				print(crop)
				cv2.rectangle(image,(x+1, y), (x + winW-1, y + winH), (255/360 * 30, 255/360 * 30, 255/360 * 30),cv2.FILLED)


			cv2.imshow("windowSize", window)


			# since we do not have a classifier, we'll just draw the window
			clone = image.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			time.sleep(0.05)
			if(crop >= 4):
				pos(vet,cp)
				break
		if(crop >= 4):
			pos(vet,cp)
			break
		# cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
