from turtle import forward
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets


class UNet(nn.Module):
    def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers) # *으로 list unpacking 

            return cbr


    def __init__(self, n_classes):
        super(UNet, self).__init__()

        # 축소
        self.enc1_1 = self.CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = self.CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = self.CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = self.CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = self.CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = self.CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = self.CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = self.CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = self.CBR2d(in_channels=512, out_channels=1024)

        # 확대
        self.dec5_1 = self.CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = self.CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = self.CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = self.CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = self.CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)


        self.dec2_2 = self.CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = self.CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_3 = self.CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_2 = self.CBR2d(in_channels=64, out_channels=64)
        self.dec1_1 = nn.Conv2d(in_channels=64, out_channels=n_classes,
                                 kernel_size=1, stride=1, padding=0,
                                 bias=True)

    def forward(self, x):
        x = self.enc1_1(x)
        enc1 = self.enc1_2(x)
        x = self.pool1(enc1)

        x = self.enc2_1(x)
        enc2 = self.enc2_2(x)
        x = self.pool2(enc2)

        x = self.enc3_1(x)
        enc3 = self.enc3_2(x)
        x = self.pool3(enc3)

        x = self.enc4_1(x)
        enc4 = self.enc4_2(x)
        x = self.pool4(enc4)

        x = self.enc5_1(x)

        x = self.dec5_1(x)

        x = self.unpool4(x)
        x = torch.cat((x, enc4), dim=1)
        x = self.dec4_2(x)
        x = self.dec4_1(x)

        x = self.unpool3(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.dec3_2(x)
        x = self.dec3_1(x)

        x = self.unpool2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.dec2_2(x)
        x = self.dec2_1(x)
			
  
        x = self.unpool1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec1_2(x)
        x = self.dec1_1(x)

        out = x

        return out