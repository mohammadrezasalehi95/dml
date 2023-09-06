import numpy as np
import os
from torch.utils.data import Dataset

class VinImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None, phase_train = True):
        data_files = list(os.walk(img_dir, topdown=False))[0][2][::-1]
        mask_files = list(os.walk(mask_dir, topdown=False))[0][2][::-1]
        self.transform=transform
        self.target_transform=transform
        filter_phase=lambda x :not 'Test' in x if phase_train  else   ('Test' in x)
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.data=list(filter(filter_phase,data_files))
        self.mask=list(filter(filter_phase,mask_files))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get paths form data[idx] and load mask and image and apply necessary transforms on it
        image = np.load(os.path.join(self.img_dir, self.data[idx]))['arr_0']
        label = np.load(os.path.join(self.mask_dir, self.mask[idx]))['arr_0']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label).float() / 255.0
        return image, label
    
    
class DDSMImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None, phase_train = True):
        data_files = list(os.walk(img_dir, topdown=False))[0][2][::-1]
        mask_files = list(os.walk(mask_dir, topdown=False))[0][2][::-1]
        self.transform=transform
        self.target_transform=transform
        filter_phase=lambda x :not 'Test' in x if phase_train  else   ('Test' in x)
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.data=list(filter(filter_phase,data_files))
        self.mask=list(filter(filter_phase,mask_files))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get paths form data[idx] and load mask and image and apply necessary transforms on it
        image = np.load(os.path.join(self.img_dir, self.data[idx]))['arr_0']
        label = np.load(os.path.join(self.mask_dir, self.mask[idx]))['arr_0']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

import itertools
from utils import os_support_path,read_csv_list
import csv
class MergedDataset(Dataset):
    def __init__(self, dir_1,dir_1_sep, dir_2,dir_2_sep,size,transform=None, target_transform=None, phase='train' ,wandb_ex=None):
        files_1=read_csv_list(os_support_path(dir_1_sep+phase+".csv"))
        files_2=read_csv_list(os_support_path(dir_2_sep+phase+".csv"))
    
        dir_1_mask=os_support_path(dir_1+"mask/"+size)
        dir_1_data=os_support_path(dir_1+"data/"+size)
        dir_2_mask=os_support_path(dir_2+"mask/"+size)
        dir_2_data=os_support_path(dir_2+"data/"+size)
        endless_iterator_2 = itertools.cycle(files_2) # second dataet must have fewer insctances
        data=[]
        self.wandb_ex=wandb_ex
        for fn1 in files_1:
            data.append({
                "mask_dir":dir_1_mask,
                "img_dir":dir_1_data,
                "file_name":fn1[0],
                "label":1
            })
            fn2=next(endless_iterator_2)
            data.append({
                "mask_dir":dir_2_mask,
                "img_dir":dir_2_data,
                "file_name":fn2[0],
                "label":0
            })
        self.transform=transform
        self.target_transform=transform
        self.data=data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data=self.data[idx]
        label=data['label']
        # get paths form data[idx] and load mask and image and apply necessary transforms on it
        image = np.load(os.path.join(data['img_dir'], data['file_name']))['arr_0']
        if label:            
            mask = np.load(os.path.join(data['mask_dir'], data['file_name']))['arr_0']    
        else:
            mask = np.load(os.path.join(data['mask_dir'], data['file_name']))['arr_0']*255
        if image.ndim == 2:
            image = image[:, :, None]
    # Ensure mask is 3D (H, W, C)
        if mask.ndim == 2:
            mask = mask[:, :, None]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, mask , label

    