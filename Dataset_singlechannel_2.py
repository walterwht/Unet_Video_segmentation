import torch
import numpy as np
from torchvision.datasets.folder import default_loader
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import os

OPsize=512

def transformdata(image, mask):
    # Random horizontal flipping
    RHF = random.random()
    
    # Random vertical flipping
    RVF = random.random()
    
    
    # Image
    resize = transforms.Resize(size=(OPsize+128))
    image = resize(image)
    
    # image Random crop
    t, l, h, w = transforms.RandomCrop.get_params(image, output_size=(OPsize, OPsize))
    image = TF.crop(image, t, l, h, w)

    #image flipping
    if RHF > 0.5:
        image = TF.hflip(image)
        
    if RVF > 0.5:
        image = TF.vflip(image)
        
    # image Grayscale
    image = transforms.Grayscale(1)(image)
    
    newMasks = torch.zeros((92,OPsize,OPsize))
    
    for mc in range(92):
         nmask = transforms.ToPILImage(mode="L")(mask[mc])
         nmask = resize(nmask)
         nmask = TF.crop(nmask, t, l, h, w)
         if RHF > 0.5:
            nmask = TF.hflip(nmask)
         if RVF > 0.5:
            nmask = TF.vflip(nmask)
         nmasknumpy = TF.to_tensor(nmask)
         newMasks[mc] += nmasknumpy[0]

    # Transform to tensor
    image = TF.to_tensor(image)
    
    return image, newMasks

class cocodataset(data.Dataset):
  def __init__(self, root, annFile, transform=None,target_transform=None):
    from pycocotools.coco import COCO
    import pycocotools._mask as coco_mask
    self.coco = COCO(annFile)
    self.ids = list(sorted(self.coco.imgs.keys()))
    self.transform=transform
    self.target_transform = target_transform
    self.root = root
    self.coco_mask=coco_mask


  def __getitem__(self, index):
    coco = self.coco
    img_id = self.ids[index]
    img_metadata = coco.loadImgs(ids=img_id)[0]
    path = img_metadata['file_name']
    img = Image.open(os.path.join(self.root, path)).convert('RGB')

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((92,img_metadata['height'],img_metadata['width']),dtype=np.uint8)

    for i in range(len(anns)):
      mask[anns[i]['category_id']] = coco.annToMask(anns[i])*255

    img, target = transformdata(img, mask)
    
    
    #fmask = torch.zeros((OPsize,OPsize),dtype=np.uint8)
    
    Tmask = np.zeros((OPsize,OPsize),dtype=np.uint)
    for m1 in range(92):
     Tmask = np.maximum(Tmask,target[0][m1]*m1)

    return img, Tmask

  def __len__(self):
    return len(self.ids)


