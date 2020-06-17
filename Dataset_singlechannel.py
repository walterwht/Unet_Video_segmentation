import torch
import numpy as np
from torchvision.datasets.folder import default_loader
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from pycocotools.coco import COCO
from PIL import Image
import os



def transformdata(image, mask):
    # Resize
    resize = transforms.Resize(size=(512+128))
    image = resize(image)
    mask = resize(mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Grayscale
    image = transforms.Grayscale(image)
    
    # Transform to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    return image, mask

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
      mask[anns[i]['category_id']] = coco.annToMask(anns[i])

    mask = transforms.ToPILImage()(mask)

    img, target = transformdata(img, mask)

    return img, target[0]

  def __len__(self):
    return len(self.ids)


