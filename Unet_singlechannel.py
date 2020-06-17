import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import models
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConvBlock(nn.Module):
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
    super().__init__()
    self.maxpool=nn.MaxPool2d(2)
    self.conv=DoubleConvBlock(in_channels, out_channels)
  
  def forward(self,x):
    x=self.maxpool(x)
    x=self.conv(x)
    return x
 
class Up(nn.Module):
  def __init__(self,in_channels,out_channes):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
    self.conv = DoubleConvBlock(in_channels, out_channels)
    
  def forward(self, x1, x2):
    x1 = self.up(x1)
    #CWH
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = doubleConvBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

