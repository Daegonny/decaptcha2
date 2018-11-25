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


def accuracy(vet):
        rope =[]
        media = []

        for i in vet:
            with open('results/accuracy'+str(i)+'.csv', 'rb') as file:
                reader = csv.reader(file)
                for line in reader:
                    if(line[0].split(';')[4] == 'percent'):
                        continue
                    else:
                        rope.append(line[0].split(';')[4])

                media.append(calcMedia(rope))
        print("Media acaracia")
        print(media)

        plt.plot( vet, media, 'go' )
        plt.plot( vet, media, 'k', color = 'blue' )
        plt.title("Media das Acuracias")
        plt.savefig('MediaAcuracias.png')
        plt.show()


def running_loss():

    rope =[]
    media = []

    with open('results/running_loss2400.csv', 'rb') as file:
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
    plt.title("Running Loss")
    plt.savefig('RunningLoss.png')
    plt.show()


def main():
    vet = ['400', '800', '1200', '1600','2000','2400' ]
    accuracy(vet)
    running_loss()

main()
