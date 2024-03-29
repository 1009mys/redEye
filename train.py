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
    
    print("===========================================")
    print("Train Start")
    print("===========================================")
    # define the image transformation for trains_ds
    # in paper, using FiveCrop, normalize, horizontal reflection
    train_transformer = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    #transforms.Grayscale(1),
                    transforms.RandomAffine(degrees=(0, 360)),
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.RandomRotation(degrees=[0,360]),
                    transforms.ColorJitter(brightness=0.2),
                    #transforms.Resize((options.resolution, options.resolution)),



    ])

    # define the image transforamtion for test0_ds
    test_transformer = transforms.Compose([
                    #transforms.Resize((options.resolution, options.resolution)),
    ])

    redEye_train = RedEye(annotations_file = data + "/annotation_train.csv", img_dir = data + '/train', transform=train_transformer)
    redEye_test  = RedEye(annotations_file = data + "/annotation_test.csv",  img_dir = data + '/test', transform=test_transformer)
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

    if modelNum=='0':
        model = efficientnet_b0(class_num)
    elif modelNum=='1':
        model = efficientnet_b1(class_num)
    elif modelNum=='2':
        model = efficientnet_b2(class_num)
    elif modelNum=='3':
        model = efficientnet_b3(class_num)
    elif modelNum=='4':
        model = efficientnet_b4(class_num)
    elif modelNum=='5':
        model = efficientnet_b5(class_num)
    elif modelNum=='6':
        model = efficientnet_b6(class_num)
    elif modelNum=='7':
        model = efficientnet_b7(class_num)

    elif modelNum=='-0' and pre_trained==1:
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=class_num)
    elif modelNum=='-1' and pre_trained==1:
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=class_num)
    elif modelNum=='-2' and pre_trained==1:
        model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=class_num)
    elif modelNum=='-3' and pre_trained==1:
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=class_num)
    elif modelNum=='-4' and pre_trained==1:
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=class_num)
    elif modelNum=='-5' and pre_trained==1:
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=class_num)
    elif modelNum=='-6' and pre_trained==1:
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=class_num)
    elif modelNum=='-7' and pre_trained==1:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=class_num)

    elif modelNum=='-0':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=class_num)
    elif modelNum=='-1':
        model = EfficientNet.from_name('efficientnet-b1', num_classes=class_num)
    elif modelNum=='-2':
        model = EfficientNet.from_name('efficientnet-b2', num_classes=class_num)
    elif modelNum=='-3':
        model = EfficientNet.from_name('efficientnet-b3', num_classes=class_num)
    elif modelNum=='-4':
        model = EfficientNet.from_name('efficientnet-b4', num_classes=class_num)
    elif modelNum=='-5':
        model = EfficientNet.from_name('efficientnet-b5', num_classes=class_num)
    elif modelNum=='-6':
        model = EfficientNet.from_name('efficientnet-b6', num_classes=class_num)
    elif modelNum=='-7':
        model = EfficientNet.from_name('efficientnet-b7', num_classes=class_num)
    elif modelNum=='VGG':
        model = VGG19_Regression()

    NGPU = torch.cuda.device_count()
    device = torch.device("cuda")

    model = nn.DataParallel(model)   # 4개의 GPU를 이용할 경우 pre_trained
    print("-------------------------")
    for i in range(NGPU):
        print(torch.cuda.get_device_name(i))
    print(sys.version)
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
   

    for epoch in range(num_epoch):
        for idx, (image, label) in enumerate(train_loader):
            model.train()
            x = image.to(device)
            x = x.float()
            #label = label.float()
            # label = list(label)
            y_ = label.to(device)

            # train데이터 셋 feedforwd 과정
            output = model.forward(x)

            #y_ = y_.view(y_.size(0), 1).float()
            #y_ = y_.float()
            # loss 계산
            loss = loss_func(output, y_)

            # optimizer 초기화 및 weight 업데이트
            optimizer.zero_grad()  # 그래디언트 제로로 만들어주는 과정
            f = loss.mean()
            loss.backward()  # backpropagation
            
            #loss.mean().backward()
            optimizer.step()

            


            

            #if idx % 100 == 0:
        print('epoch : ', epoch)
        #print('loss : ', loss.data)
        

        model.eval()
        test_loss = 0
        correct = 0
        
        loss_func

        guesses = np.array([])
        labels = np.array([])

        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
                #target = target.float()
                data=data.float()
                output = model(data)
                #print(output, target)
                #target = target.view(target.size(0), 1).float()
                lossT = loss_func(output, target)
                test_loss +=  lossT.item()
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                tmp1 = np.array(pred.to('cpu'))
                tmp2 = np.array(target.to('cpu'))

                tt1 = np.array(tmp1[:])
                tt2 = np.array(tmp2[:])

                guesses = np.append(guesses, tt1)
                labels = np.append(labels, tt2)

                #print(len(guesses))
                

        #guesses = guesses.astype(int)
        labels = labels.astype(int)
        guesses = guesses.astype(int)

        guesses = list(guesses)
        labels = list(labels)

        print(guesses,'\n',labels)

        print(classification_report(labels, guesses, labels=range(0, class_num)))

        #misc (acc 계산, etc) 
        acc = accuracy_score(labels, guesses)
        f_score = f1_score(labels, guesses, average='macro')


        if acc > best_acc:
            best_acc = acc
            best_acc_model = deepcopy(model.state_dict())

            with open('./result/' + modelNum + '_' + result_name + "_best_acc.txt", "w") as text_file:
                print("epoch:", epoch, file=text_file)
                print("train loss:", test_loss, file=text_file)
                print(classification_report(labels, guesses, digits=3), file=text_file)
            
            torch.save(best_acc_model, './result/' + modelNum + '_' + result_name + '_best_acc.pt')

        if f_score > best_f1:
            best_f1 = f_score
            best_f1_model = deepcopy(model.state_dict())
            
            with open('./result/' + modelNum + '_' + result_name + "_best_f1.txt", "w") as text_file:
                print("epoch:", epoch, file=text_file)
                print("train loss:", test_loss, file=text_file)
                print(classification_report(labels, guesses, digits=3), file=text_file)

            torch.save(best_f1_model, './result/' + modelNum + '_' + result_name + '_best_f1.pt')

        #print('accuracy:', round(accuracy_score(labels, guesses), ndigits=3))
        #print('recall score:', round(recall_score(labels, guesses, average='micro'), ndigits=3))
        #print('precision score:', round(precision_score(labels, guesses, average='micro'), ndigits=3))
        #print('f1 score:', round(f1_score(labels, guesses, average='micro'), ndigits=3))

        print('-----------------')
    
        

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--batch", "-b", default=1, dest="batch_size", type=int)
    parser.add_option("--learning_rate", "-l", default=0.0001, dest="learning_rate", type=float)
    parser.add_option("--epoch", "-e", default=500, dest="num_epoch", type=int)
    parser.add_option("--model", "-1", default='0', dest="model", type=str)
    parser.add_option("--workers", "-w", default=1, dest="workers", type=int)
    parser.add_option("--class_num", "-c", default=11, dest="class_num", type=int)
    parser.add_option("--data", "-d", default="./data/redEye", dest="data", type=str)
    parser.add_option("--result_name", "-n", default="", dest="result_name", type=str)
    parser.add_option("--loss", default="criterion", dest="loss_function", type=str)
    parser.add_option("--pre_trained", default="0", dest="pre_trained", type=int)
    parser.add_option("--weight", "-W", default="", dest="weight", type=str)
    parser.add_option("--resolution", "-r", default=224, dest="resolution", type=int)

    

    

    trainEffNet(parser)
