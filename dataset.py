import torch
import numpy as np
from torch.utils.data import Dataset
import cv2 as cv
import os


class USDataset(Dataset):
    def __init__(self,root,filename,transform=None,mode='train'):
        self.root = root+mode+"/"
        self.transform = transform
        self.mode = mode
        self.filenames = filename

    def __getitem__(self, item):
        filename = self.filenames[item]
        image = cv.imread(os.path.join(self.root+"data/",filename))
        label = cv.imread(os.path.join(self.root + "gt/", filename), 0)
        # image,label = self.randomHorizontalFlip(image,label)
        image = image.transpose((2,0,1))
        label = label[np.newaxis, :, :]
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image,label


    def __len__(self):
        return len(self.filenames)

    def randomHorizontalFlip(self,img,mask,u=0.5):
        if np.random.random() < u:
            img = cv.flip(img,1)
            mask = cv.flip(mask,1)
        return img,mask
