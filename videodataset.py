import torch
import numpy as np
from torchvision.datasets.folder import default_loader
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from inputvideo import*


class videodataset(data.Dataset):
  def __init__(self, root,time,size):
    self.root = root
    self.time = time
    self.size = size
    
    


  def __getitem__(self, index):
    video = read_video(self.root,3,self.time,self.size)
    frame = video[index]    
    return frame

  def __len__(self):
    return self.time

