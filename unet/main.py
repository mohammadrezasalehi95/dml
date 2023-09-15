

from utils import get_epoch_from_filename, load_recentliest_file
from utils import os_support_path
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import random
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from datasets import CoupledDataset
import wandb
import time
import numpy as np
from multiprocessing import cpu_count
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import module
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch
from torch import Tensor
from utils import balanced_focal_cross_entropy_loss
import torchvision.models as models
from evaluate import *

import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(12345)
torch.cuda.manual_seed_all(12345)
# same seed on all devices; both CPU and CUDA
torch.manual_seed(1345)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed_all(12345)


"""#Model"""
"""# Utils"""

"""#Model"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return F.sigmoid(logits)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


"""#Data sets"""


class DDSMImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None, phase_train=True):
        data_files = list(os.walk(img_dir, topdown=False))[0][2][::-1]
        mask_files = list(os.walk(mask_dir, topdown=False))[0][2][::-1]
        self.transform = transform
        self.target_transform = transform
        def filter_phase(
            x): return not 'Test' in x if phase_train else ('Test' in x)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.data = list(filter(filter_phase, data_files))
        self.mask = list(filter(filter_phase, mask_files))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get paths form data[idx] and load mask and image and apply necessary transforms on it
        image = np.load(os.path.join(self.img_dir, self.data[idx]))['arr_0']
        label = np.load(os.path.join(self.mask_dir, self.mask[idx]))['arr_0']
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)*255
        return image, label


# my_transforms = transforms.Compose([transforms.ToTensor()])


def collate_fn(batch):
    inpput_src, input_tr, gt_src, gt_tr = zip(*batch)
    inpput_src = [torch.from_numpy(np.squeeze(arr)) for arr in inpput_src]
    inpput_src = torch.stack(inpput_src)

    input_tr = [torch.from_numpy(np.squeeze(arr)) for arr in input_tr]
    input_tr = torch.stack(input_tr)

    gt_src = [torch.from_numpy(np.squeeze(arr)) for arr in gt_src]
    gt_src = torch.stack(gt_src)

    gt_tr = [torch.from_numpy(np.squeeze(arr)) for arr in gt_tr]
    gt_tr = torch.stack(gt_tr)
    return inpput_src.unsqueeze(1), input_tr.unsqueeze(1), gt_src.unsqueeze(1), gt_tr.unsqueeze(1)


def train_model(
        model,
        device,
        current_epoch=1,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        dir_checkpoint=None,
):
    train_dataset = CoupledDataset(dir_1="Vin/",
                                   dir_1_sep="Vin/sep/V1/",
                                   size="512/",
                                   dir_2="DDSM/",
                                   dir_2_sep="DDSM/sep/ORG/",
                                   # transform=target_tr,
                                   # target_transform=target_tr,
                                   phase='train',
                                   # wandb_ex=experiment,
                                   fix_to_first=True,
                                   )
    val_dataset = CoupledDataset(dir_1="Vin/",
                                 dir_1_sep="Vin/sep/V1/",
                                 size="512/",
                                 dir_2="DDSM/",
                                 dir_2_sep="DDSM/sep/ORG/",
                                 # transform=target_tr,
                                 # target_transform=target_tr,
                                 phase='val',
                                 # wandb_ex=experiment,
                                 fix_to_first=False,
                                 )
    val_showing_dataset = CoupledDataset(dir_1="Vin/",
                                         dir_1_sep="Vin/sep/V1/",
                                         size="512/",
                                         dir_2="DDSM/",
                                         dir_2_sep="DDSM/sep/ORG/",
                                         # transform=target_tr,
                                         # target_transform=target_tr,
                                         phase='val2'
                                         # wandb_ex=experiment,
                                         )
    test_dataset = CoupledDataset(dir_1="Vin/",
                                  dir_1_sep="Vin/sep/V1/",
                                  size="512/",
                                  dir_2="DDSM/",
                                  dir_2_sep="DDSM/sep/ORG/",
                                  # transform=target_tr,
                                  # target_transform=target_tr,
                                  phase='test'
                                  # wandb_ex=experiment,
                                  )

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    loader_args = dict(batch_size=batch_size,
                        collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, shuffle=False, **loader_args,num_workers=5)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, shuffle=False, drop_last=True, **loader_args,num_workers=1)
    val_showing_loader = torch.utils.data.DataLoader(
        dataset=val_showing_dataset, shuffle=False, **loader_args,num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, shuffle=False, drop_last=True, **loader_args,num_workers=1)

    # (Initialize logging)
    experiment = wandb.init(project='Unet',
                            # mode="offline"
                            )
    experiment.config.update(
        dict(current_epoch=current_epoch, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(current_epoch, current_epoch+epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{current_epoch+epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, gb, true_masks, gb_mask = batch
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(
                    device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = balanced_focal_cross_entropy_loss(
                            masks_pred, true_masks, focal_gamma=2).mean()
                    else:
                        return 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # Evaluation round
                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' +
                                           tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' +
                                           tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score,eval = evaluate(
                            model, val_loader, device, amp)
                        # scheduler.step(val_score)
                        logging.info(
                            'Validation Dice score: {}'.format(val_score))
                        model.eval()
                        source = {}
                        target = {}
                        step_showing = 0
                        for batch in val_showing_loader:
                            images_src, images_tr, mask_src, mask_tr = batch
                            # pmax=lambda a:np.min(a.numpy())
                            # print("dddddddddddddddddd",pmax(images_src),pmax(images_tr),pmax(mask_src),pmax(mask_tr))
                            images_src = images_src.to(
                                device=device, dtype=torch.float32, memory_format=torch.channels_last)
                            images_tr = images_tr.to(
                                device=device, dtype=torch.float32, memory_format=torch.channels_last)
                            pmasks_src = model(images_src)
                            pmasks_tr = model(images_tr)
                            images_src=images_src.cpu()
                            pmasks_src7=(pmasks_src >= 0.7).float().cpu()
                            pmasks_src=(pmasks_src >= 0.5).float().cpu()
                            images_tr=images_tr.cpu()
                            pmasks_tr7=(pmasks_tr >= 0.7).float().cpu()
                            pmasks_tr=(pmasks_tr >= 0.5).float().cpu()
                            listing_image=lambda images:[wandb.Image(transforms.ToPILImage()(image), caption="An example image") for image in images]
                            source.update({
                                f'image': listing_image(images_src),
                                f'mask': listing_image(mask_src.float()),
                                f'pmask5': listing_image(pmasks_src),
                                f'pmask7': listing_image(pmasks_src7),
                            })
                            target.update({
                                f't_image': listing_image(images_tr),
                                f't_mask': listing_image(mask_tr.float()),
                                f't_pmask5': listing_image(pmasks_tr),
                                f't_pmask7': listing_image(pmasks_tr7),
                            })
                            break
                            # step_showing+=batch_size
                        experiment.log({
                            'validation Dice': val_score,
                            '_tp':eval._tp_per_class[1],
                            '_fp':eval._fp_per_class[1],
                            '_fn':eval._fn_per_class[1],
                            'source': source,
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
                        model.train()
        if dir_checkpoint:

            state_dict = model.state_dict()
            torch.save(state_dict, os_support_path(
                str(dir_checkpoint + f'checkpoint_epoch{epoch}.pth')))
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    class Temp:
        def __init__(self):
            pass

    args = Temp()
    args.load = True
    args.amp = False
    args.classes = 1
    args.scale = 1
    args.batch_size = 4
    args.bilinear = False
    args.epochs = 10
    args.dir_checkpoint = "Model/unet/"
    args.lr = 1e-5
    args.amp = False

    torch.cuda.device_count()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=1)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        file = load_recentliest_file(args.dir_checkpoint)
        if file:
            current_epoch = get_epoch_from_filename(file)+1
            state_dict = torch.load(file, map_location=device)
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {file}')
        else:
            current_epoch = 1

    model.to(device=device)
    train_model(
        model=model,
        current_epoch=current_epoch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        amp=args.amp,
        dir_checkpoint=args.dir_checkpoint,
    )
