from email.policy import default
import sys, getopt
from optparse import OptionParser
from xmlrpc.client import boolean

from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.nn as nn # layer들을 호출하기 위해서
import torch.optim as optim # optimization method를 사용하기 위해서
import torch.nn.init as init # weight initialization 해주기 위해서
import torchvision.datasets as dset # toy data들을 이용하기 위해서
import torchvision.transforms as transforms # pytorch 모델을 위한 데이터 변환을 위해
from torch.utils.data import DataLoader # train,test 데이터를 loader객체로 만들어주기 위해서

from dataLoader import RedEye
from model_effNet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from model_uNet import UNet
from model_VGG19 import VGG19_Regression

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

from copy import deepcopy

from efficientnet_pytorch import EfficientNet




def trainEffNet(parser):

    (options, args) = parser.parse_args()

    batch_size = options.batch_size
    learning_rate = options.learning_rate
    num_epoch = options.num_epoch
    modelNum = options.model
    class_num = options.class_num
    data = options.data
    result_name = options.result_name
    loss_function = options.loss_function
    pre_trained = options.pre_trained
    weight = options.weight

    print("===========================================")
    print("test Start")
    print("===========================================")
    # define the image transformation for trains_ds
    # in paper, using FiveCrop, normalize, horizontal reflection
    train_transformer = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    #transforms.Grayscale(1),
                    transforms.RandomAffine(degrees=(0, 360))



    ])

    # define the image transforamtion for test0_ds
    test_transformer = transforms.Compose([
                    
    ])

    redEye_train = RedEye(annotations_file = data + "/annotation_train.csv", img_dir = data + '/train', transform=train_transformer)
    redEye_test  = RedEye(annotations_file = data + "/annotation_test.csv",  img_dir = data + '/test')
    """
    redEye_train = dset.ImageNet("./data/ImageNet",
                               train=True,
                               transform=transforms.ToTensor(),  # torch안에서 연산을 해주기 위한 형태로 변환
                               target_transform=None,
                               download=True)
    redEye_test = dset.ImageNet("./data/ImageNet",
                              train=False,
                              transform=transforms.ToTensor(),  # torch안에서 연산을 해주기 위한 형태로 변환
                              target_transform=None,
                              download=True)
    
    redEye_train = dset.CIFAR10("./data/CIFAR10",
                               train=True,
                               transform=transforms.ToTensor(),  # torch안에서 연산을 해주기 위한 형태로 변환
                               target_transform=None,
                               download=True)
    redEye_test = dset.CIFAR10("./data/CIFAR10",
                              train=False,
                              transform=transforms.ToTensor(),  # torch안에서 연산을 해주기 위한 형태로 변환
                              target_transform=None,
                              download=True)
    """

    # Data loader 객체 생성
    # 데이터 batch로 나눠주고 shuffle해주는 데이터 loader 객체 생성
    
    train_loader = DataLoader(redEye_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=options.workers,
                              drop_last=True,
                              pin_memory=True
                              )
    test_loader = DataLoader(redEye_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=options.workers,
                             drop_last=False,
                             pin_memory=True
                             )
    
   
    

    model = None

    if modelNum=='VGG':
        model = VGG19_Regression()

    NGPU = torch.cuda.device_count()
    device = torch.device("cuda")

    model = nn.DataParallel(model)   # 4개의 GPU를 이용할 경우 pre_trained

    model.load_state_dict(torch.load(weight))

    print("-------------------------")
    for i in range(NGPU):
        print(torch.cuda.get_device_name(i))
    print("-------------------------")
    model.to(device)

    loss_func = None
    if loss_function == 'criterion':
        loss_func = nn.CrossEntropyLoss()  # 크로스엔트로피 loss 객체, softmax를 포함함
    elif loss_function == 'MSE':
        loss_func = nn.MSELoss()
    else:
        raise Exception("올바른 loss함수가 아님!")

    

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    loss_list = []
    acc_list = []
    best_acc = 0
    best_f1 = 0
    best_acc_model = None 
    best_f1_model = None
   

    
        

    model.eval()
    test_loss = 0
    correct = 0
    
    loss_func

    guesses = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
            target = target.float()
            data=data.float()
            output = model(data)
            
            target = target.view(target.size()[0], 1)

            #print(output, target)
            lossT = loss_func(output, target)
            test_loss +=  lossT.item()
            #pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

            tmp1 = np.array(output.to('cpu'))
            tmp2 = np.array(target.to('cpu'))

            #print("pred", output)
            #print("target", target)

            tt1 = np.array(tmp1[:])
            tt2 = np.array(tmp2[:])

            print(tt1)

            guesses = np.append(guesses, tt1)
            labels = np.append(labels, tt2)

            #print(len(guesses))

    #guesses = guesses.astype(int)
    labels = labels.astype(int)

    guesses = list(guesses)
    labels = list(labels)

    #print('guesses, labels', guesses,'\n',labels)

    print("test loss = ", test_loss / batch_size)

    #print("guesses: ", guesses)
    #print("===============================================")
    """
    for i in range(len(guesses)):
        if float(guesses[i]) - int(guesses[i]) > 0.5:
            guesses[i] = int(guesses[i])+1
            
        else:
            guesses[i] = int(guesses[i])

        #print(i)
    """
    #print("===============================================")

    

    print('guesses, labels', guesses,'\n',labels)

    print(classification_report(labels, guesses, labels=[0,1,2,3,4,5,6,7,8,9,10]))

    #misc (acc 계산, etc) 
    acc = accuracy_score(labels, guesses)
    f_score = f1_score(labels, guesses, average='macro')


    
        

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--batch", "-b", default=16, dest="batch_size", type=int)
    parser.add_option("--learning_rate", "-l", default=0.0001, dest="learning_rate", type=float)
    parser.add_option("--epoch", "-e", default=500, dest="num_epoch", type=int)
    parser.add_option("--model", "-m", default='VGG', dest="model", type=str)
    parser.add_option("--workers", "-w", default=1, dest="workers", type=int)
    parser.add_option("--class_num", "-c", default=11, dest="class_num", type=int)
    parser.add_option("--data", "-d", default="./data/redEye", dest="data", type=str)
    parser.add_option("--result_name", "-n", default="", dest="result_name", type=str)
    parser.add_option("--loss", default="MSE", dest="loss_function", type=str)
    parser.add_option("--pre_trained", default="0", dest="pre_trained", type=int)
    parser.add_option("--weight", "-W", default="", dest="weight", type=str)

    

    

    trainEffNet(parser)
