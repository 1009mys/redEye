from email.policy import default
import sys, getopt
from optparse import OptionParser

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
from model_effNet import EfficientNet, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b7
from model_uNet import UNet

def trainEffNet(parser):

    (options, args) = parser.parse_args()

    batch_size = int(options.batch_size)
    learning_rate = int(options.learning_rate)
    num_epoch = int(options.num_epoch)

    print("===========================================")
    print(batch_size, learning_rate, num_epoch)
    print("===========================================")
    # define the image transformation for trains_ds
    # in paper, using FiveCrop, normalize, horizontal reflection
    train_transformer = transforms.Compose([
                    transforms.RandomHorizontalFlip()

    ])

    # define the image transforamtion for test0_ds
    test_transformer = transforms.Compose([
                    
    ])

    redEye_train = RedEye(annotations_file="./data/redEye/annotation_train.csv", img_dir='./data/redEye/train', transform=train_transformer)
    redEye_test  = RedEye(annotations_file="./data/redEye/annotation_test.csv", img_dir='./data/redEye/test')
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
                              num_workers=1,
                              drop_last=True)
    test_loader = DataLoader(redEye_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1,
                             drop_last=True)
    
   
    device = torch.device("cuda")

    model = nn.DataParallel(efficientnet_b7(11), device_ids = [0,1,2,3])   # 4개의 GPU를 이용할 경우

    model.to(device)

    loss_func = nn.CrossEntropyLoss()  # 크로스엔트로피 loss 객체, softmax를 포함함
    optimizer = optim.Adam(model.parameters(),  # 만든 모델의 파라미터를 넣어줘야 함
                           lr=learning_rate)

   

    for epoch in range(num_epoch):
        for idx, (image, label) in enumerate(train_loader):
            model.train()
            x = image.to(device)
            x = x.float()
            # label = list(label)
            y_ = label.to(device)

            # train데이터 셋 feedforwd 과정
            output = model.forward(x)

            # loss 계산
            loss = loss_func(output, y_)

            # optimizer 초기화 및 weight 업데이트
            optimizer.zero_grad()  # 그래디언트 제로로 만들어주는 과정
            loss.backward()  # backpropagation
            optimizer.step()

            if idx % 100 == 0:
                print('epoch : ', epoch)
                print('loss : ', loss.data)
                

                model.eval()
                test_loss = 0
                correct = 0
                criterion =  nn.CrossEntropyLoss(reduction='sum')

                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
                        data=data.float()
                        output = model(data)
                        #print(output, target)
                        loss = criterion(output, target)
                        test_loss +=  loss.item()
                        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(test_loader.dataset)

                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

                print('-----------------')

if __name__ == "__main__":
    batch_size = 8
    learning_rate = 0.001
    num_epoch = 500

    parser = OptionParser()
    parser.add_option("--batch", "-b", default=8, dest="batch_size")
    parser.add_option("--learning_rate", "-l", default=0.001, dest="learning_rate")
    parser.add_option("--epoch", "-e", default=500, dest="num_epoch")
    

    

    trainEffNet(parser)
