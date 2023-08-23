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


class MergedDataset(Dataset):
    def __init__(self, img_dir_1, mask_dir_1, img_dir_2, mask_dir_2,transform=None, target_transform=None, phase = "Train"):
        
        data_files_1 = list(os.walk(img_dir_1, topdown=False))[0][2][::-1]
        mask_files_1 = list(os.walk(mask_dir_1, topdown=False))[0][2][::-1]
        data_files_2 = list(os.walk(img_dir_2, topdown=False))[0][2][::-1]
        mask_files_2 = list(os.walk(mask_dir_2, topdown=False))[0][2][::-1]
        endless_iterator_2 = itertools.cycle(data_files_2)
        data=[]
        for fn1 in data_files_1:
            data.append({
                "mask_dir":mask_dir_1,
                "img_dir":img_dir_1,
                "file_name":fn1,
                "label":1
            })
            fn2=next(endless_iterator_2)
            data.append({
                "mask_dir":mask_dir_2,
                "img_dir":img_dir_2,
                "file_name":fn2,
                "label":0
            })
            
        
            
        self.transform=transform
        self.target_transform=transform
        filter_phase=lambda x :not 'Test' in x if phase_train  else   ('Test' in x)
        
        self.data=data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data=self.data[idx]
        # get paths form data[idx] and load mask and image and apply necessary transforms on it
        image = np.load(os.path.join(data['img_dir'], data['file_name']))['arr_0']
        mask = np.load(os.path.join(data['mask_dir'], data['file_name']))['arr_0']
        label=data['label']
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

    