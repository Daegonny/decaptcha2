# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
import numpy as np

def calcMedia(rope):
    x=0
    for i in rope:
        x= x + float(i)

    return x/len(rope)

def calcLoss(rope):
    rope = np.asarray(rope)
    rope = rope.reshape(7200,1)
    rope = rope.reshape(30,240)
    sum = rope.sum(axis=0, dtype='float')
    media = sum/30
    return media

def calcLossTxT(rope):
    rope = np.asarray(rope)
    rope = rope.reshape(2700,1)
    rope = rope.reshape(30,90)
    sum = rope.sum(axis=0, dtype='float')
    media = sum/30
    return media


# def accuracy(vet):
#         rope =[]
#         media = []
#
#         for i in vet:
#             with open('slide_window/results/accuracy'+str(i)+'.csv', 'rb') as file:
#                 reader = csv.reader(file)
#                 for line in reader:
#                     if(line[0].split(';')[4] == 'percent'):
#                         continue
#                     else:
#                         rope.append(line[0].split(';')[4])
#
#                 media.append(calcMedia(rope))
#         print("Media acaracia")
#         print(media)
#
#         plt.plot( vet, media, 'go' )
#         plt.plot( vet, media, 'k', color = 'blue' )
#         plt.title("Media das Acuracias")
#         plt.savefig('MediaAcuracias.png')
#         plt.show()


def running_loss():

    rope =[]
    media = []

    with open('slide_window/results/running_loss2400.csv', 'rb') as file:
    #with open('test.csv', 'rb') as file:
        reader = csv.reader(file)
        for line in reader:
            if(line[0].split(';')[2] == 'metric'):
                continue
            else:
                rope.append(float(line[0].split(';')[2]))

        media = calcLoss(rope)

    vet = np.arange(1,241)
    print("RunningLoss")
    print(media)

    plt.plot( vet, media, 'go' )
    plt.plot( vet, media, 'k', color = 'blue' )
    plt.title("Running Loss - CNN Reconhecedora I")
    plt.savefig('Graphics/RunningLossCNNvI.png')
    plt.show()

    rope =[]
    media = []

    with open('slide_window/results/running_loss2400-2.csv', 'rb') as file:
    #with open('test.csv', 'rb') as file:
        reader = csv.reader(file)
        for line in reader:
            if(line[0].split(';')[2] == 'metric'):
                continue
            else:
                rope.append(float(line[0].split(';')[2]))

        media = calcLoss(rope)

    vet = np.arange(1,241)
    print("RunningLoss")
    print(media)

    plt.plot( vet, media, 'go' )
    plt.plot( vet, media, 'k', color = 'blue' )
    plt.title("Running Loss - CNN Reconhecedora II")
    plt.savefig('Graphics/RunningLossCNNvII.png')
    plt.show()

    rope =[]
    media = []

    with open('slide_window/results/running_loss_text_non_text_2400.csv', 'rb') as file:
    #with open('test.csv', 'rb') as file:
        reader = csv.reader(file)
        for line in reader:
            if(line[0].split(';')[2] == 'metric'):
                continue
            else:
                rope.append(float(line[0].split(';')[2]))

        media = calcLossTxT(rope)

    vet = np.arange(1,91)
    print("RunningLoss")
    print(media)

    plt.plot( vet, media, 'go' )
    plt.plot( vet, media, 'k', color = 'blue' )
    plt.title('Running Loss - CNN Detec'+ u'\u00E7'+r'$\tilde{a}$o de Falha')
    plt.savefig("Graphics/RunningLossTextNonText.png")
    plt.show()


def main():
    vet = ['2400' ]
    #accuracy(vet)
    running_loss()

main()
