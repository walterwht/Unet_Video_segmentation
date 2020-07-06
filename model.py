import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, padding=0, kernel_size=1, stride=1, with_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.with_relu = with_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x

class FinalStage(nn.Module):
    def __init__(self, input_channel, output_channel, padding=0, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        #self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        #x = self.relu(x)
        #x = self.sm(x)
        return x


class Resnet50Unet(nn.Module):
    def __init__(self, y_classes=92):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.Encoderlayers = list(self.model.children())

        self.inputlayer = nn.Sequential(*self.Encoderlayers[:3])
        self.layer1 = nn.Sequential(*self.Encoderlayers[3:5])
        self.layer2 = self.Encoderlayers[5]
        self.layer3 = self.Encoderlayers[6]
        self.layer4 = self.Encoderlayers[7]
        self.fullConnet = self.Encoderlayers[8]

        self.uplayer1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.uplayer2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.uplayer3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.uplayer4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.uplayer5 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        self.upconv1 = ConvBlock(2048, 1024)
        self.upconv2 = ConvBlock(1024, 512)
        self.upconv3 = ConvBlock(512, 128)
        self.upconv4 = ConvBlock(128, 64)
        self.upconv5 = ConvBlock(64, 64)

        #self.UnetClasses = nn.Conv2d(64, y_classes, 1)

        self.FinalStage = FinalStage(64,y_classes)

    def forward(self, x):
        downlayer1 = self.inputlayer(x)
        downlayer2 = self.layer1(downlayer1)
        downlayer3 = self.layer2(downlayer2)
        downlayer4 = self.layer3(downlayer3)
        out = self.layer4(downlayer4)

        out = self.uplayer1(out)
        out = torch.cat((out, downlayer4), dim=1)
        out = self.upconv1(out)

        out = self.uplayer2(out)
        out = torch.cat((out, downlayer3), dim=1)
        out = self.upconv2(out)

        out = self.uplayer3(out)
        out = torch.cat((out, downlayer2), dim=1)
        out = self.upconv3(out)

        out = self.uplayer4(out)
        out = torch.cat((out, downlayer1), dim=1)
        out = self.upconv4(out)

        out = self.uplayer5(out)

        #out = self.UnetClasses(out)
        out = self.FinalStage(out)

        return out


