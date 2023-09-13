import time

from torch.nn.utils.rnn import pad_sequence
import logging
from pathlib import Path
import torch
import torch.nn as nn
from utils import Eval_FP
from utils import get_epoch_from_filename, load_recentliest_file

import wandb
from torch import Tensor
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random

from torchvision import transforms
from tqdm import tqdm
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from utils import balanced_focal_cross_entropy_loss
import torchvision.models as models
from parts import *
from evaluate import *
from datasets import *

from torchvision import datasets, transforms, models
import itertools
import numpy as np


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed_all(12345)


"""#Model"""


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
        self.discre_start = (DiscreStart(1536, 32))
        self.CRCR1 = (CRCR(32, 64))
        self.CRCR2 = (CRCR(64, 128))
        self.CRCR3 = (CRCR(128, 256))
        self.CRCR4 = (CRCR(256, 512))
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

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
        v = self.discre_start(x4, x5)
        v = self.CRCR1(v)
        v = self.CRCR2(v)
        v = self.CRCR3(v)
        v = self.CRCR4(v)
        v = v.view(v.size(0), -1)  # flatten v
        v = self.fc(v)
        v = self.sigmoid(v)

        return F.sigmoid(logits), v

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

def collate_fn_couple(batch):
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

def collate_fn_merge(batch):
    images, masks, labels = zip(*batch)
    images = [torch.from_numpy(np.squeeze(arr)).unsqueeze(0) for arr in images]
    images = torch.stack(images)
    masks = [torch.from_numpy(np.squeeze(arr)).unsqueeze(0) for arr in masks]
    masks = torch.stack(masks)

    if isinstance(labels[0], torch.Tensor):
        if labels[0].shape == torch.Size([]):  # labels are scalar
            labels = torch.stack(labels, 0)
        else:  # labels are sequences
            labels = pad_sequence(labels, batch_first=True)
    else:
        # Convert labels to tensor if they are not
        labels = torch.tensor(labels)
    return images, masks, labels
# Use the new collate function in DataLoader


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

    experiment = wandb.init(project='unet-vgg',
                            # mode="disabled",
                            # name="vgg_unet-with merged dataset"
                            )

    data_transforms = transforms.Compose([transforms.ToTensor()])
    target_tr = transforms.Compose([transforms.ToTensor()])
    train_dataset = MergedDataset(dir_1="Vin/",
                                  dir_1_sep="Vin/sep/V1/",
                                  size="512/",
                                  dir_2="DDSM/",
                                  dir_2_sep="DDSM/sep/ORG/",
                                  transform=target_tr,
                                  target_transform=target_tr,
                                  phase='train'
                                  # wandb_ex=experiment,
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

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    loader_args = dict(batch_size=batch_size,)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False,
                                               num_workers=5,
                                               **loader_args, collate_fn=collate_fn_merge)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             num_workers=1,
                                             shuffle=False, drop_last=True, **loader_args, collate_fn=collate_fn_couple)
    val_showing_loader = torch.utils.data.DataLoader(
        dataset=val_showing_dataset, shuffle=False, **loader_args, num_workers=1, collate_fn=collate_fn_couple)

    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             #  img_scale=img_scale,
             amp=amp)
    )

    # logging.info(f'''Starting training:
    #     Epochs:          {epochs}=
    #     Batch size:      {batch_size}
    #     Learning rate:   {learning_rate}
    #     Training size:   {n_train}
    #     Validation size: {n_val}5.078179597854614
    #     Device:          {device.type}
    #     Images scaling:  {img_scale}
    #     Mixed Precision: {amp}
    # ''')
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
    criterion2 = nn.BCELoss()
    global_step = 0
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks, labels = batch
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                images = images.to(device=device, dtype=torch.float32,
                                   memory_format=torch.channels_last)  # todo float check
                true_masks = true_masks.to(device=device, dtype=torch.long)

                labels = labels.to(device=device, dtype=torch.float32)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred, labels_pred = model(images)
                    # if model.n_classes == 1:
                    # loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    B, H, W = labels.shape[0], 512, 512
                    # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    ignore_mask = labels.view(B, 1, 1, 1).expand(-1, 1, H, W)

                    loss1 = balanced_focal_cross_entropy_loss(probs=masks_pred,
                                                              gt=true_masks,
                                                              ignore_mask=ignore_mask,
                                                              focal_gamma=2
                                                              ).mean()

                    loss2 = balanced_focal_cross_entropy_loss(probs=labels_pred.unsqueeze(2).unsqueeze(3),
                                                              gt=labels.unsqueeze(
                                                                  1).unsqueeze(2).unsqueeze(3),
                                                              focal_gamma=2
                                                              ).mean()/50
                loss = loss1+loss2
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss_dir': loss1.item(),
                    'train loss_adv_source': loss2.item(),
                    'step': global_step,
                    'epoch': epoch,
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # Evaluation round
                division_step = (n_train // (1112*batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' +
                                           tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' +
                                           tag] = wandb.Histogram(value.grad.data.cpu())
                        val_score, junk = evaluate(
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
                            images_src = images_src.cpu()
                            pmasks_src7 = (pmasks_src >= 0.7).float().cpu()
                            pmasks_src = (pmasks_src >= 0.5).float().cpu()
                            images_tr = images_tr.cpu()
                            pmasks_tr7 = (pmasks_tr >= 0.7).float().cpu()
                            pmasks_tr = (pmasks_tr >= 0.5).float().cpu()
                            def listing_image(images): return [wandb.Image(transforms.ToPILImage()(
                                image), caption="An example image") for image in images]
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
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'source': source,
                            'target': target,
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
    args.batch_size = 10
    args.bilinear = False
    args.epochs = 10
    args.dir_checkpoint = "Model/vgg_unet/"
    args.lr = 1e-5
    args.amp = False

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
            logging.info(f'Model loaded from {args.load}')
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
