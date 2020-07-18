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
        self.conv = nn.Conv2d(input_channel, input_channel, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(input_channel, output_channel, padding=padding, kernel_size=kernel_size, stride=stride)

        #self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        
        #x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        #x = self.sm(x)
        return x


class Resnet50Unet(nn.Module):
    def __init__(self, y_classes=92):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=False, progress=True,num_classes=21, aux_loss=None)
        self.Resnet = list(self.model.children())
        self.Encoderlayers = list(self.Resnet[0].children())
        self.FCNhead = list(self.Resnet[1].children())
        for p in self.parameters():
            p.requires_grad = False

        self.inputlayer = nn.Sequential(*self.Encoderlayers[:3])
        self.layer1 = nn.Sequential(*self.Encoderlayers[3:5])
        self.layer2 = self.Encoderlayers[5]
        self.layer3 = self.Encoderlayers[6]
        self.layer4 = self.Encoderlayers[7]
        self.fullConnet = self.FCNhead[0]
        print(self.Encoderlayers)
        print(self.fullConnet)


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.uplayer1 = nn.ConvTranspose2d(2048, 1024, 1, stride=1)
        self.uplayer2 = nn.ConvTranspose2d(1024, 512, 1, stride=1)
        self.uplayer3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.uplayer4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.upconv1 = ConvBlock(2048, 1024)
        self.upconv2 = ConvBlock(1024, 512)
        self.upconv3 = ConvBlock(512, 128)
        self.upconv4 = ConvBlock(128, 64)
        
        self.conv_original_size0 = ConvBlock(3, 64, 1, 3)
        self.conv_original_size1 = ConvBlock(64, 64, 1, 3)
        self.conv_original_size2 = ConvBlock(128, 64, 1, 3)
        
        self.FinalStage = FinalStage(64,y_classes)

    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        inputlayer = self.inputlayer(x)#64 channel [2, 64, 128, 128])
        downlayer1 = self.layer1(inputlayer)# 256 channel [2, 256, 64, 64]
        downlayer2 = self.layer2(downlayer1)# 512 channel [2, 512, 32, 32]
        downlayer3 = self.layer3(downlayer2)# 1024 channel[2, 1024, 32, 32]
        downlayer4 = self.layer4(downlayer3)# 2048 channel[2, 2048, 32, 32]
        
        out = self.uplayer1(downlayer4)# 2048 -> 1024       
        out = torch.cat((out, downlayer3), dim=1)# 1024+1024
        out = self.upconv1(out)# 2048 -> 1024
        
        out = self.uplayer2(out)# 1024 -> 512
        out = torch.cat((out, downlayer2), dim=1) # 512 + 512
        out = self.upconv2(out)# 1024 -> 512
        
        out = self.uplayer3(out) # 512 -> 256
        out = torch.cat((out, downlayer1), dim=1)# 256 + 256
        out = self.upconv3(out)# 512 -> 128
        
        out = self.uplayer4(out) # 128 -> 64
        out = torch.cat((out, inputlayer), dim=1)# 64 + 64
        out = self.upconv4(out)# 128 -> 64
        
        out = self.upsample(out)
        out = torch.cat([out, x_original], dim=1)
        
        out = self.conv_original_size2(out)     
 
        #out = self.UnetClasses(out)
        out = self.FinalStage(out)
        return out


