import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.utils.data as data



class SampleProperty(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def length(self):
        return self._data[1]

    @property
    def label(self):
        return int(self._data[2])

def Getimagesname(samplepath):
    imagesname=[]
    numstr_num={}
    if samplepath.find('/CASME/')!=-1:
        for image in os.listdir(samplepath):
            l = image.find('-')
            e = image.find('.')
            head = image[0:l+1]
            numstr_num[image[l+1:e]] = int(image[l+1:e])
            end = '.jpg'
        numstr_num = sorted(numstr_num.items(),key=lambda x:x[1])
        for t in numstr_num:
            imagesname.append(head+t[0]+end)
        return imagesname
    if samplepath.find('/CASMEII/')!=-1:
        for image in os.listdir(samplepath):
            l = image.find('img')
            e = image.find('.')
            head = image[0:l+3]
            numstr_num[image[l+3:e]] = int(image[l+3:e])
            end = '.jpg'
        numstr_num = sorted(numstr_num.items(),key=lambda x:x[1])
        for t in numstr_num:
            imagesname.append(head+t[0]+end)
        return imagesname
    if samplepath.find('/SAMM/')!=-1:
        for image in os.listdir(samplepath):
            l = image.find('_')
            e = image.find('.')
            head = image[0:l+1]
            numstr_num[image[l+1:e]] = int(image[l+1:e])
            end = '.jpg'
        numstr_num = sorted(numstr_num.items(),key=lambda x:x[1])
        for t in numstr_num:
            imagesname.append(head+t[0]+end)
        return imagesname
    if samplepath.find('/SMIC/')!=-1:
        for image in os.listdir(samplepath):
            l = image.find('image')
            e = image.find('.')
            head = 'image'
            numstr_num[image[l+5:e]] = int(image[l+5:e])
            end = '.bmp'
        numstr_num = sorted(numstr_num.items(),key=lambda x:x[1])
        for t in numstr_num:
            imagesname.append(head+t[0]+end)
        return imagesname



class VolumeDataset(data.Dataset):
    def __init__(self, data_root, list_file_root, modality='Gray', transform=None):
        self.data_root = data_root
        self.list_file_root = list_file_root
        self.modality = modality
        self.transform = transform
        self._images_load()
    def _images_load(self):
        self.Sample_List = [SampleProperty(x.strip().split(' ')) for x in open(self.list_file_root)]

    def __getitem__(self, idx):
        sample = self.Sample_List[idx]

        Volume_temp = list()
        imagesname = Getimagesname(sample.path)
        for i in imagesname:
            if self.modality == 'RGB':
                image = Image.open(os.path.join(self.data_root, sample.path, i)).convert('RGB')
            if self.modality == 'Gray':
                image = Image.open(os.path.join(self.data_root, sample.path, i)).convert('L')
            Volume_temp.append(image)

        if self.transform is not None:
            Volume = self.transform(Volume_temp)   ###C L H W
      
        SampleVolum = {'Volume': Volume, 'label': sample.label}
        return SampleVolum

    def __len__(self):
        return len(self.Sample_List)