import numpy as np
import pandas as pd
import torch
import torch.utils.data
import os
import random
import torchvision
from PIL import Image
import random

from torchvision import transforms

class med(torch.utils.data.Dataset):
    def __init__(self,df,transforms = None):
        img_list = df['imgpath']
        label = df['label_num']
        self.ziplist = list(zip(img_list,label))
        self.transform = transforms

    def __getitem__(self, index):
        img_path,label = self.ziplist[index]
        img_path = img_path.replace('datab','datah')
        imgi = Image.open(img_path)
        imgi = self.transform(imgi)
        return imgi,label

    def __len__(self):
        return len(self.ziplist)

class med_infer(torch.utils.data.Dataset):
    def __init__(self,df,transforms = None):
        img_list = df['imgpath']
        self.img_list = list(img_list)
        self.transform = transforms

    def __getitem__(self,index):
        img_path = self.img_list[index]
        imgi = Image.open(img_path)
        imgi = self.transform(imgi)
        return imgi,img_path

    def __len__(self):
        return len(self.img_list)

'''
batch_size = 10
device = 'cuda'

trans = transforms.ToTensor()

custom_dataset = med('final_filelist.csv',transforms = trans)
train_dataset = torch.utils.data.DataLoader(dataset=custom_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
for i, (imagei,label,img_path) in enumerate(train_dataset):
    print(img_path,label)
    img = imagei.to(device)
'''
