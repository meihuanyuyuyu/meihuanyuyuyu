import matplotlib.pyplot as plt
import json
from matplotlib.pyplot import MultipleLocator
import os

from scipy.ndimage.measurements import label

def pict(i,dtype:str,*args):
    y0 = []
    y1 =[]
    y2= []
    x1 = json.load(open(os.path.join(args[0],f'_{i}_1_train_score.json'),'r'))
    x2 = json.load(open(os.path.join(args[1],f'_{i}_1_train_score.json'),'r'))
    x3 = json.load(open(os.path.join(args[2],f'_{i}_1_train_score.json'),'r'))
    for index,data in enumerate(x1[dtype]):
        if index%5 == 0:
            y0.append(data)
    for index,data in enumerate(x2[dtype]):
        if index%5 == 0:
            y1.append(data)
    for index,data in enumerate(x3[dtype]):
        if index%5 == 0:
            y2.append(data)
    plt.plot(y0,color='r',label='attention u-net')
    plt.plot(y1,color='g',label='residual u-net')
    plt.plot(y2,color='b',label='u-net')
    y_major_locator= MultipleLocator(0.1)
    plt.xlabel('every 5 epochs')
    plt.ylabel('loss')
    plt.title('model training loss')
    plt.ylim(0.1,1.0)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()

def sub_pic(pos:int,i,dtype:str,*args:str):
    y0 = []
    y1 =[]
    y2= []
    x1 = json.load(open(os.path.join(args[0],f'_{i}_0_train_score.json'),'r'))
    x2 = json.load(open(os.path.join(args[1],f'_{i}_0_train_score.json'),'r'))
    x3 = json.load(open(os.path.join(args[2],f'_{i}_0_train_score.json'),'r'))
    for index,data in enumerate(x1[dtype]):
        if index%5 == 0:
            y0.append(data)
    for index,data in enumerate(x2[dtype]):
        if index%5 == 0:
            y1.append(data)
    for index,data in enumerate(x3[dtype]):
        if index%5 == 0:
            y2.append(data)
    plt.subplot(5,3,pos)
    plt.plot(y0,color='r',label='attention u-net',)
    plt.plot(y1,color='g',label='residual u-net')
    plt.plot(y2,color='b',label='u-net')
    if dtype == 'loss':
        plt.xlabel('every 5 epochs')
        plt.ylabel('loss')
        plt.title('model training loss')
        plt.ylim(0.1,0.8)
    if dtype =='f1':
        plt.xlabel('every 5 epochs')
        plt.ylabel('F1 score')
        plt.title('model training F1 Score')

    if dtype =='val_f1':
        plt.xlabel('every 5 epochs')
        plt.ylabel('val F1 score')
        plt.title('model val set F1 Score')
    
def multi_pic(*args):
    fig = plt.figure(figsize=[14.4,25.6])
    for i in range(1,16):
        if i % 3 ==1:
            n = 0
            sub_pic(i,n,'loss','result/EMstacks/attention json','result/EMstacks/R2 u-net','result/EMstacks/unet_json')
            n +=1
        if i %3 ==2:
            j =0
            sub_pic(i,j,'f1','result/EMstacks/attention json','result/EMstacks/R2 u-net','result/EMstacks/unet_json')
            n +=1
        if i %3 ==0:
            k = 0
            sub_pic(i,k,'val_f1','result/EMstacks/attention json','result/EMstacks/R2 u-net','result/EMstacks/unet_json')
            k +=1
    plt.suptitle('different model with 5-fold validation')
    plt.legend()
    fig.tight_layout()
    plt.show()

def three_pic(*args):
    fig = plt.figure()
    for i in range(1,4):
        sub_pic(i,0,'loss','result/EMstacks/attention json','result/EMstacks/R2 u-net','result/EMstacks/unet_json')
        sub_pic(i,0,'f1','result/EMstacks/attention json','result/EMstacks/R2 u-net','result/EMstacks/unet_json')
        sub_pic(i,0,'val_f1','result/EMstacks/attention json','result/EMstacks/R2 u-net','result/EMstacks/unet_json')
    plt.show()



# pict(1,'loss','2020seg_result/training_json/attention unet','2020seg_result/training_json/residual unet','2020seg_result/training_json/unet')
# multi_pic('result/EMstacks/attention json','result/EMstacks/R2 u-net','result/EMstacks/unet_json')
# three_pic('result/EMstacks/attention json','result/EMstacks/R2 u-net','result/EMstacks/unet_json')