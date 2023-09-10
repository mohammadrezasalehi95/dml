import time

from torch.nn.utils.rnn import pad_sequence
import logging
from pathlib import Path
import torch
import torch.nn as nn
from utils import Eval_FP

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
        self.discre_start=(DiscreStart(1536,32))
        self.CRCR1=(CRCR(32,64))
        self.CRCR2=(CRCR(64,128))
        self.CRCR3=(CRCR(128,256))
        self.CRCR4=(CRCR(256,512))
        self.fc=nn.Linear(512, 1)
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
        v=self.discre_start(x4,x5)
        v=self.CRCR1(v)
        v=self.CRCR2(v)
        v=self.CRCR3(v)
        v=self.CRCR4(v)
        v = v.view(v.size(0), -1)  # flatten v
        v = self.fc(v)
        v = self.sigmoid(v)

        return logits,v

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



    


def collate_fn(batch):
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
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        # val_percent: float = 0.15,
        save_checkpoint: bool = False,
        dir_checkpoint:str='',
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    
    experiment = wandb.init(project='Breast-vgg',
                            # mode="disabled",
                            
                            # name="vgg_unet-with merged dataset"
                            )

    data_transforms = transforms.Compose([transforms.ToTensor()])
    target_tr = transforms.Compose([transforms.ToTensor()])
    train_dataset=MergedDataset( dir_1="Vin/", 
                                dir_1_sep="Vin/sep/V1/",
                                size="512/",
                                dir_2="DDSM/", 
                                dir_2_sep="DDSM/sep/ORG/",
                                transform=target_tr,
                                target_transform=target_tr,
                                phase='train'
                                # wandb_ex=experiment,
                                )
    val_dataset=MergedDataset( dir_1="Vin/", 
                                dir_1_sep="Vin/sep/V1/",
                                size="512/",
                                dir_2="DDSM/", 
                                dir_2_sep="DDSM/sep/ORG/",
                                transform=target_tr,
                                target_transform=target_tr,
                                phase='val'
                                # wandb_ex=experiment,
                                )
        
    n_train = len(train_dataset)
    n_val = len(val_dataset) 
    
    loader_args = dict(batch_size=batch_size, num_workers=1, )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=False, **loader_args,collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,shuffle=False,drop_last=True, **loader_args ,collate_fn=collate_fn)
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint,
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
    real_tensor=torch.tensor(np.zeros((batch_size,1))).float().to(device=device)
    fake_tensor=torch.tensor(np.ones((batch_size,1))).float().to(device=device)

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images,true_masks,labels=batch
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last) # todo float check
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                labels=labels.to(device=device,dtype=torch.float32 )
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred,labels_pred = model(images)
                    # if model.n_classes == 1:
                    # loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    B,H,W=labels.shape[0],512,512
                    # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    ignore_mask = labels.view(B, 1, 1, 1).expand(-1, 1, H, W)
                    
                    loss1 = balanced_focal_cross_entropy_loss(probs=F.sigmoid(masks_pred),
                                                              gt=true_masks ,
                                                              ignore_mask=ignore_mask,
                                                              focal_gamma=2
                                                              ).mean()
                        
                        

                    loss2= balanced_focal_cross_entropy_loss(probs=labels_pred.unsqueeze(2).unsqueeze(3),
                                                              gt= labels.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                                                              focal_gamma=2
                                                              ).mean()/50
                        
                
                loss=loss1+loss2
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
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
                division_step = (n_train // (2*batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        val_score,junk = evaluate(model, val_loader, device, amp)
                        # scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred05': wandb.Image((F.sigmoid(masks_pred[0])>=0.5).float().cpu()),
                                    'pred07': wandb.Image((F.sigmoid(masks_pred[0])>=0.7).float().cpu()),
                                    'real_fake':labels_pred[0].float().cpu()
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        if save_checkpoint:
            Path(os_support_path(dir_checkpoint)).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, os_support_path(str(dir_checkpoint + '/checkpoint_epoch{}.pth'.format(epoch))))
            logging.info(f'Checkpoint {epoch} saved!')
# do stuff


if __name__ == '__main__':    

    # img_dir=os_support_path("Vin/data/512")
    # dir_img = Path(os_support_path('DDSM/data/512/'))
    # dir_mask = Path(os_support_path('DDSM/mask/512/'))
    # dir_checkpoint = Path(os_support_path('checkpoints/'))
    class Temp:
      def __init__(self):
        pass

    args=Temp()
    args.load = 'Model/vgg_unet/checkpoint_epoch10.pth'
    args.amp=False
    args.classes=1
    args.scale=1
    args.batch_size=2
    args.bilinear=False
    args.epochs=10
    args.lr=2e-5
    args.val=15
    args.amp=False
    args.dir_checkpoint='Model/vgg_unet/02/'
    args.save_checkpoint=True

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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
        state_dict = torch.load(os_support_path(args.load), map_location=device)
        if 'mask_values' in state_dict.keys():
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    

    
    val_dataset=MergedDataset( dir_1="Vin/", 
                                dir_1_sep="Vin/sep/V1/",
                                size="512/",
                                dir_2="DDSM/", 
                                dir_2_sep="DDSM/sep/ORG/",
                                # transform=target_tr,
                                # target_transform=target_tr,
                                phase='val'
                                # wandb_ex=experiment,
                                )
        
    n_val = len(val_dataset) 
    
    loader_args = dict(batch_size=16, num_workers=1, )
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,shuffle=False,drop_last=True, **loader_args ,collate_fn=collate_fn)
    junk,eval=evaluate(model, val_loader, device,args.amp)
    # model.eval()
    # eval=Eval_FP()
    # for batch in val_loader:
        
    #     images,true_masks,labels=batch
    #     assert images.shape[1] == model.n_channels, \
    #         f'Network has been defined with {model.n_channels} input channels, ' \
    #         f'but loaded images have {images.shape[1]} channels. Please check that ' \
    #         'the images are loaded correctly.'
    #     images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last) # todo float check
    #     true_masks = true_masks.to(device=device, dtype=torch.long)
    #     labels=labels.to(device=device,dtype=torch.float32 )
    #     masks_pred,labels_pred = model(images)
    #     eval.eval(masks_pred.detach().cpu().numpy(),true_masks.cpu().numpy())
    # model.train()
    

        
    
    print("fn",eval._fn_per_class,"\nfp",eval._fp_per_class,"\ntp",eval._tp_per_class,"\nn_sample",eval._n_received_samples)
        
        

    # train_model(
    #     model=model,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     learning_rate=args.lr,
    #     device=device,
    #     img_scale=args.scale,
    #     amp=args.amp,
    #     save_checkpoint=args.save_checkpoint,
    #     dir_checkpoint=args.dir_checkpoint,
    # )
