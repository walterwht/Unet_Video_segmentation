import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import models
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class doubleConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, padding=0, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Down(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().-_init__()
    maxpool=nn.MaxPool2d(2)
    conv=DoubleConv(in_channels, out_channels)
  
  def forward(self,x):
    x=maxpool(x)
    x=conv(x)
    return x
 
class Up(nn.Module):
  def __init__(self


class Unet(nn.Module):
    def __init__(self, y_classes=92):
        super().__init__()
        self.inputlayer = doubleConvBlock(1,64)
        self.layer2 = doubleConvBlock(64,128)
        self.layer3 = doubleConvBlock(128,256)
        self.layer4 = doubleConvBlock(256,512)
        self.layer5 = doubleConvBlock(512,1024)


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


