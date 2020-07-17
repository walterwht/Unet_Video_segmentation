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


OPsize=520


def transformdata(image, mask):
    # Random horizontal flipping
    RHF = random.random()
    
    # Random vertical flipping
    RVF = random.random()
    
    
    # Image
    resize = transforms.Resize(size=(OPsize+64))
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
    #image = transforms.Grayscale(1)(image)
    
    newMasks = torch.zeros((21,OPsize,OPsize))
    
    for mc in range(len(newMasks)):
         nmask = transforms.ToPILImage(mode="L")(mask[mc]*255)
         nmask = resize(nmask)
         nmask = TF.crop(nmask, t, l, h, w)
         if RHF > 0.5:
            nmask = TF.hflip(nmask)
         if RVF > 0.5:
            nmask = TF.vflip(nmask)
         nmasknumpy = TF.to_tensor(nmask)
         newMasks[mc] += nmasknumpy[0].round()

    # Transform to tensor
    image = TF.to_tensor(image)
    image = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(image)
    newMasks[0] = 1-newMasks[1:][0]
    
    
    return image, newMasks

class cocodataset(data.Dataset):
  def __init__(self, root, annFile, classes):
    from pycocotools.coco import COCO
    import pycocotools._mask as coco_mask
    self.coco = COCO(annFile)
    self.ids = len(classes)
    self.root = root
    self.coco_mask=coco_mask
    self.classes = classes
    


  def __getitem__(self, index):
    allclassnms = self.classes
    coco = self.coco        
    catIds = coco.getCatIds(catNms=self.ids[index])
    imgIds = coco.getImgIds(catIds=catIds)
    imgid = imgIds[np.random.randint(0, len(imgIds))]
    img_metadata = coco.loadImgs(imgIds=imgid)[0]
    path = img_metadata['file_name']
    img = Image.open(os.path.join(self.root, path)).convert('RGB')

    ann_ids = coco.getAnnIds(imgIds=imgid)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((len(allclassnms),img_metadata['height'],img_metadata['width']),dtype=np.uint8)
    
    annIds = coco.getAnnIds(imgIds=imgid, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    
    
    for i in range(len(anns)):
        for cnms in allclassnms:
            cat_id = anns[i]["category_id"]
            if cat_id == cnms:
                catname = coco.loadCats(cat_id)
                classnumber = allclassnms.index(catname[0]['name'])
                mask[classnumber] += coco.annToMask(anns[i])

    inimg, target = transformdata(img, mask)
    

      
    return inimg, target

  def __len__(self):
    return len(self.classes)

