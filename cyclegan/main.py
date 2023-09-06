import os
import json
import random
from datasets import MergedDataset
from utils import os_support_path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision.utils import save_image
import losses

# Assuming the necessary modules are defined in the following files
# import data_loader, losses, model
# from stats_func import *

# Set the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(config_filename):
    # Set the seed for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    sifa_model = SIFA(config)
    sifa_model.train()
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



class SIFA:
    def __init__(self, config):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # self._source_train_pth = config['source_train_pth']
        # self._target_train_pth = config['target_train_pth']
        # self._source_val_pth = config['source_val_pth']
        # self._target_val_pth = config['target_val_pth']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._pool_size = int(config['pool_size'])
        self._lambda_a = float(config['_LAMBDA_A'])
        self._lambda_b = float(config['_LAMBDA_B'])
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])
        self._base_lr = float(config['base_lr'])
        self._max_step = int(config['max_step'])
        self._keep_rate_value = float(config['keep_rate_value'])
        self._is_training_value = bool(config['is_training_value'])
        self._batch_size = int(config['batch_size'])
        self._lr_gan_decay = bool(config['lr_gan_decay'])
        self._to_restore = bool(config['to_restore'])
        self._checkpoint_dir = config['checkpoint_dir']

        # Define the models
        # self.model_A = model.ModelA().to(device)
        # self.model_B = model.ModelB().to(device)
        # self.discriminator = model.Discriminator().to(device)

        # Define the optimizers
        self.optimizer_A = optim.Adam(self.model_A.parameters(), lr=self._base_lr)
        self.optimizer_B = optim.Adam(self.model_B.parameters(), lr=self._base_lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self._base_lr)

        # Define the loss functions
        self.criterion_GAN = losses.GANLoss().to(device)
        self.criterion_cycle = losses.CycleConsistencyLoss().to(device)
        self.criterion_identity = losses.IdentityLoss().to(device)
        
        # Load datasets
        # self.dataloader_source_train = DataLoader(data_loader.CustomDataset(self._source_train_pth),
        #                                           batch_size=self._batch_size, shuffle=True)
        # self.dataloader_target_train = DataLoader(data_loader.CustomDataset(self._target_train_pth),
        #                                           batch_size=self._batch_size, shuffle=True)
    def train(self):
        # Training loop
        train_dataset=MergedDataset( dir_1="Vin/", 
                                dir_1_sep="Vin/sep/V1/",
                                size="512/",
                                dir_2="DDSM/", 
                                dir_2_sep="DDSM/sep/ORG/",
                                # transform=target_tr,
                                # target_transform=target_tr,
                                phase='train'
                                # wandb_ex=experiment,
                                )
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
            
        n_train = len(train_dataset)
        n_val = len(val_dataset) 
        
        loader_args = dict(batch_size=self._batch_size, num_workers=1, )
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=False, **loader_args,collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,shuffle=False,drop_last=True, **loader_args ,collate_fn=collate_fn)
        
        
        for epoch in range(self._max_step):
            for i, (images, true_masks, true_labels) in enumerate(train_loader):
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last) # todo float check
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_labels=true_labels.to(device=device,dtype=torch.float32 )
                
                # Transfer data to the appropriate device
                # Update generator_A and generator_B
                self.optimizer_A.zero_grad()
                self.optimizer_B.zero_grad()
                # You should replace the following lines with your actual model training code
                # loss_A, loss_B = train_generator(self.model_A, self.model_B, real_A, real_B, self.criterion_GAN, self.criterion_cycle, self.criterion_identity)
                # loss_A.backward()
                # loss_B.backward()
                # self.optimizer_A.step()
                # self.optimizer_B.step()

                # Update discriminator
                # self.optimizer_D.zero_grad()
                # loss_D = train_discriminator(self.discriminator, real_A, real_B, self.model_A, self.model_B, self.criterion_GAN)
                # loss_D.backward()
                # self.optimizer_D.step()

                # Print losses and save models
                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Iter: {i}, Loss Generator A: {loss_A.item()}, Loss Generator B: {loss_B.item()}, Loss Discriminator: {loss_D.item()}")

                if i % 1000 == 0:
                    # Save model checkpoints
                    torch.save(self.model_A.state_dict(), os.path.join(self._checkpoint_dir, f'model_A_{epoch}_{i}.pth'))
                    torch.save(self.model_B.state_dict(), os.path.join(self._checkpoint_dir, f'model_B_{epoch}_{i}.pth'))
                    torch.save(self.discriminator.state_dict(), os.path.join(self._checkpoint_dir, f'discriminator_{epoch}_{i}.pth'))

                    # Save sample images
                    with torch.no_grad():
                        self.model_A.eval()
                        self.model_B.eval()
                        fake_A = self.model_A(real_B)
                        fake_B = self.model_B(real_A)
                        save_image(fake_A, os.path.join(self._images_dir, f'fake_A_{epoch}_{i}.png'))
                        save_image(fake_B, os.path.join(self._images_dir, f'fake_B_{epoch}_{i}.png'))
                        self.model_A.train()
                        self.model_B.train()

if __name__ == '__main__':
    main(config_filename=os_support_path('cyclegan/config_param.json'))