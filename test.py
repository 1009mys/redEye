
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
from model_effNet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from model_uNet import UNet

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import normalize

from copy import deepcopy

from efficientnet_pytorch import EfficientNet

def testEffNet(parser):
    (options, args) = parser.parse_args()

    batch_size = options.batch_size
    learning_rate = options.learning_rate
    num_epoch = options.num_epoch
    modelNum = options.model
    class_num = options.class_num
    data = options.data
    result_name = options.result_name
    weight = options.weight

    redEye_test  = RedEye(annotations_file = data + "/annotation_test.csv",  img_dir = data + '/test')

    test_loader = DataLoader(redEye_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=options.workers,
                             drop_last=False,
                             pin_memory=True
                             )

    model = None
    
    if modelNum=='-0':
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
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(weight))

    model = model.module
    device = torch.device("cuda")
    model.to(device)

    model.eval()

    test_loss = 0
    correct = 0
    criterion =  nn.CrossEntropyLoss(reduction='sum')

    guesses = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
            data=data.float()
            output = model(data)
            #print(output, target)
            lossT = criterion(output, target)
            test_loss +=  lossT.item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            tmp1 = np.array(pred.to('cpu'))
            tmp2 = np.array(target.to('cpu'))

            tt1 = np.array(tmp1[:])
            tt2 = np.array(tmp2[:])

            guesses = np.append(guesses, tt1)
            labels = np.append(labels, tt2)


            features = model.extract_features(data)

            features_ = features.to('cpu').tolist()


            print(len(features_[0]))

            r = None

            for idxx, ff in enumerate(features_[0]):
                
                if r == None:
                    r = ff

                else:

                    r = (np.array(r) + np.array(ff)).tolist()

            r = np.array(r)

            if idx == 65:
                print(idx, r)

            #print(len(guesses))

    guesses = guesses.astype(int)
    labels = labels.astype(int)

    guesses = list(guesses)
    labels = list(labels)

    print(guesses,'\n',labels)

    print(classification_report(labels, guesses, labels=[0,1,2,3,4,5,6,7,8,9,10]))

    

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--batch", "-b", default=1, dest="batch_size", type=int)
    parser.add_option("--learning_rate", "-l", default=0.0001, dest="learning_rate", type=float)
    parser.add_option("--epoch", "-e", default=500, dest="num_epoch", type=int)
    parser.add_option("--model", "-m", default='-1', dest="model", type=str)
    parser.add_option("--workers", "-w", default=1, dest="workers", type=int)
    parser.add_option("--class_num", "-c", default=11, dest="class_num", type=int)
    parser.add_option("--data", "-d", default="./data/redEye", dest="data", type=str)
    parser.add_option("--result_name", "-n", default="", dest="result_name", type=str)
    parser.add_option("--weight", "-W", default="-1_noFilter_best_f1.pt", dest="weight", type=str)

    

    testEffNet(parser)
