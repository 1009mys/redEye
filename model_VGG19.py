# import package

# model
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import time
import copy

import torchsummary

"""
class floor(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, 
                out_channels=out_channel, 
                kernel_size=kernel_size, 
                stride=stride
            )
        )
"""


class VGG19_Regression(nn.Module):
    def __init__(self): # 0 ~ 11
        super().__init__()

        #channnels =     [64, 128, 256, 512, 512, 512, 4096, 1]
        #repeats =       [2, 3, 4, 4, 4, 1, 2, 1]
        #kernel_size =   [3, 3, 3, 3, 3, 3]

        self.floor1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.floor2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.floor3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.floor4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.floor5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.floor6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.Dropout(),
            nn.Linear(4096, 1)
        )
    def forward(self, x):
        x = self.floor1(x)
        x = self.floor2(x)
        x = self.floor3(x)
        x = self.floor4(x)
        x = self.floor5(x)
        x = self.floor6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(16, 3, 224, 224).to(device)

    model = VGG19_Regression().to(device)
    output = model(x)
    print(output.size())
    torchsummary.summary(model, (3, 224, 224))

    
   