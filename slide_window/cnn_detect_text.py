#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Nov 20 12:15:04 2018

@author: daegonny
"""
import torch
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#import torchvision
#import matplotlib.pyplot as plt
#import numpy as np
from random import shuffle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


wd = os.getcwd()
os.chdir(wd)

from slide_window.loader_detect_text import DriveData

result = []
total_running_loss = []
total_accuracy = []

for n in range(1):
    print("n = "+str(n))
    #define 2/3 = train, 1/3 = test
    all_idx = []
    train_idxs = []
    test_idxs = []
    dset_train = None
    dset_test = None
    trainloader = None
    testloader = None


    all_idx = [i for i in range(2225)]
    shuffle(all_idx)

    train_idxs = all_idx[0:1470]
    test_idxs = all_idx[1470:2225]

    

    #select train data
    dset_train = DriveData(folder_dataset="", transform=None, line_idxs=train_idxs)
    print(len(dset_train))
    #select test data
    dset_test = DriveData(folder_dataset="", transform=None, line_idxs=test_idxs)
    print(len(dset_test))
    

    trainloader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=2)
    testloader = DataLoader(dset_test, batch_size=4, shuffle=False, num_workers=2)

    classes = ('0', '1')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 5, 1)
            self.conv2 = nn.Conv2d(6, 16, 5, 1)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 2)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # If the size is a square you can only specify a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


    net = Net()
    #print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.85)

    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #if i % 100 == 99:    # print every 100 mini-batches
            #    total_running_loss.append((n,running_loss/100))
            #    running_loss = 0.0
        total_running_loss.append(running_loss/500)
        
    print('Finished Training')
    


    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    total_accuracy.append((n,correct,total,correct/total))


    torch.save(net, "slide_window/models/model.pt")

#    class_correct = list(0. for i in range(28))
#    class_total = list(0. for i in range(28))
#    with torch.no_grad():
#        for data in testloader:
#            images, labels = data
#            outputs = net(images)
#            _, predicted = torch.max(outputs, 1)
#            c = (predicted == labels).squeeze()
#            for i in range(4):
#                label = labels[i]
#                class_correct[label] += c[i].item()
#                class_total[label] += 1
#
#    for i in range(28):
#        if class_total[i] == 0:
#            pass
#        #    print('Accuracy of %5s : Not Tested' % (
#        #            classes[i]))
#        else:
#            result.append((n,classes[i], class_correct[i],class_total[i], class_correct[i] / class_total[i]))
#            #print('Accuracy of %5s : %2d / %2d = %2d %%' % (
#            #        classes[i], class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))
#
#
#
#
#
#result = pd.DataFrame(result)
#result.columns = ['n','label','hit','total','percent']
#
#total_running_loss = pd.DataFrame(total_running_loss)
#total_running_loss.columns = ['n', 'metric']
#
#total_accuracy = pd.DataFrame(total_accuracy)
#total_accuracy.columns = ['n', 'hit', 'total','percent']
#
#
#result.to_csv("results/result2400.csv", sep=';')
#total_running_loss.to_csv("results/running_loss2400.csv", sep=";")
#total_accuracy.to_csv("results/accuracy2400.csv", sep=";")
