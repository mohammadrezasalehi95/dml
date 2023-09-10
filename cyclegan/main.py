import torch.nn.functional as F
import os
import json
import random
from cyclegan.model import SIFAModule
from cyclegan.stats_func import dice_eval
from datasets import MergedDataset
from utils import os_support_path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision.utils import save_image
import losses
import wandb
import itertools
from tqdm import tqdm
# Assuming the necessary modules are defined in the following files
# import data_loader, losses, model
# from stats_func import *

# Set the device for computation


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
    def __init__(self, config, device):
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
        self.optimizer_A = optim.Adam(
            self.model_A.parameters(), lr=self._base_lr)
        self.optimizer_B = optim.Adam(
            self.model_B.parameters(), lr=self._base_lr)
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=self._base_lr)

        # Define the loss functions
        self.criterion_GAN = losses.GANLoss().to(device)
        self.criterion_cycle = losses.CycleConsistencyLoss().to(device)
        self.criterion_identity = losses.IdentityLoss().to(device)
        self.device = device
        # Load datasets
        # self.dataloader_source_train = DataLoader(data_loader.CustomDataset(self._source_train_pth),
        #                                           batch_size=self._batch_size, shuffle=True)
        # self.dataloader_target_train = DataLoader(data_loader.CustomDataset(self._target_train_pth),
        #                                           batch_size=self._batch_size, shuffle=True)

    def model_setup(self):
        self.discriminator_A_vars = [
            param for name, param in self.model.named_parameters() if 'discriminator_A' in name]
        self.discriminator_B_vars = [
            param for name, param in self.model.named_parameters() if 'discriminator_B' in name]
        self.generator_A_vars = [
            param for name, param in self.model.named_parameters() if 'generator_A' in name]
        self.encoder_B_vars = [
            param for name, param in self.model.named_parameters() if 'encoder_B' in name]
        self.decoder_B_vars = [
            param for name, param in self.model.named_parameters() if 'decoder_B' in name]
        self.segmenter_B_vars = [
            param for name, param in self.model.named_parameters() if 'segmenter_B' in name]
        self.segmenter_B_ll_vars = [
            param for name, param in self.model.named_parameters() if 'segmenter_B_ll' in name]
        self.discriminator_P_vars = [
            param for name, param in self.model.named_parameters() if 'discriminator_P' in name]
        self.discriminator_P_ll_vars = [
            param for name, param in self.model.named_parameters() if 'discriminator_P_ll' in name]

    def compute_losses(self,
                       input_a,
                       input_b,
                       cycle_images_a,
                       cycle_images_b,
                       prob_fake_a_is_real,
                       prob_fake_b_is_real,
                       prob_pred_mask_b_is_real,
                       prob_pred_mask_b_ll_is_real,
                       prob_fake_a_aux_is_real,
                       pred_mask_fake_b,
                       pred_mask_fake_b_ll,
                       gt_a,
                       prob_real_a_is_real,
                       prob_fake_pool_a_is_real,
                       prob_cycle_a_aux_is_real,
                       prob_fake_pool_a_aux_is_real,
                       prob_real_b_is_real,
                       prob_fake_pool_b_is_real,
                       prob_pred_mask_fake_b_is_real,
                       prob_pred_mask_fake_b_ll_is_real,
                       _lambda_a,
                       _lambda_b,
                       ):

        cycle_consistency_loss_a = _lambda_a * losses.cycle_consistency_loss(
            real_images=input_a, generated_images=cycle_images_a,
        )
        cycle_consistency_loss_b = _lambda_b * losses.cycle_consistency_loss(
            real_images=input_b, generated_images=cycle_images_b,
        )
        lsgan_loss_a = losses.lsgan_loss_generator(prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(prob_fake_b_is_real)
        lsgan_loss_p = losses.lsgan_loss_generator(prob_pred_mask_b_is_real)
        lsgan_loss_p_ll = losses.lsgan_loss_generator(
            prob_pred_mask_b_ll_is_real)
        lsgan_loss_a_aux = losses.lsgan_loss_generator(prob_fake_a_aux_is_real)

        ce_loss_b, dice_loss_b = losses.task_loss(pred_mask_fake_b, gt_a)
        ce_loss_b_ll, dice_loss_b_ll = losses.task_loss(
            pred_mask_fake_b_ll, gt_a)
        l2_loss_b = torch.sum([0.0001 * torch.sum(v ** 2)
                               for v in [self.segmentor_B_vars, self.encoder_B_vars, self.segmenter_B_ll_vars]])

        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        seg_loss_B = ce_loss_b + dice_loss_b + l2_loss_b + 0.1 * \
            (ce_loss_b_ll + dice_loss_b_ll) + 0.1 * g_loss_B + 0.1 * \
            lsgan_loss_p + 0.01 * lsgan_loss_p_ll + 0.1 * lsgan_loss_a_aux

        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=prob_real_a_is_real,
            prob_fake_is_real=prob_fake_pool_a_is_real,
        )
        d_loss_A_aux = losses.lsgan_loss_discriminator(
            prob_real_is_real=prob_cycle_a_aux_is_real,
            prob_fake_is_real=prob_fake_pool_a_aux_is_real,
        )
        d_loss_A = d_loss_A + d_loss_A_aux
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=prob_real_b_is_real,
            prob_fake_is_real=prob_fake_pool_b_is_real,
        )
        d_loss_P = losses.lsgan_loss_discriminator(
            prob_real_is_real=prob_pred_mask_fake_b_is_real,
            prob_fake_is_real=prob_pred_mask_b_is_real,
        )
        d_loss_P_ll = losses.lsgan_loss_discriminator(
            prob_real_is_real=prob_pred_mask_fake_b_ll_is_real,
            prob_fake_is_real=prob_pred_mask_b_ll_is_real,
        )

        return g_loss_A, g_loss_B, seg_loss_B, d_loss_A, d_loss_B, d_loss_P, d_loss_P_ll

    def train(self, epochs):
        # Training loop
        train_dataset = MergedDataset(dir_1="Vin/",
                                      dir_1_sep="Vin/sep/V1/",
                                      size="512/",
                                      dir_2="DDSM/",
                                      dir_2_sep="DDSM/sep/ORG/",
                                      # transform=target_tr,
                                      # target_transform=target_tr,
                                      phase='train'
                                      # wandb_ex=experiment,
                                      )
        val_dataset = MergedDataset(dir_1="Vin/",
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
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, shuffle=False, **loader_args, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, shuffle=False, drop_last=True, **loader_args, collate_fn=collate_fn)

        model = SIFAModule().to(self.device)


        discriminator_A_optimizer = torch.optim.Adam(self.discriminator_A_vars, lr=self.learning_rate_gan, betas=(0.5, 0.999))
        discriminator_B_optimizer = torch.optim.Adam(self.discriminator_B_vars, lr=self.learning_rate_gan, betas=(0.5, 0.999))
        generator_A_optimizer = torch.optim.Adam(self.generator_A_vars, lr=self.learning_rate_gan, betas=(0.5, 0.999))
        generator_B_optimizer = torch.optim.Adam(self.decoder_B_vars, lr=self.learning_rate_gan, betas=(0.5, 0.999))
        discriminator_P_optimizer = torch.optim.Adam(self.discriminator_P_vars, lr=self.learning_rate_gan, betas=(0.5, 0.999))
        discriminator_P_ll_optimizer = torch.optim.Adam(self.discriminator_P_ll_vars, lr=self.learning_rate_gan, betas=(0.5, 0.999))
        segmenter_B_optimizer = torch.optim.Adam(itertools.chain(self.encoder_B_vars,
                                                   self.segmenter_B_vars,
                                                   self.segmenter_B_ll_vars,
                                                   ), lr=self.learning_rate_seg)
        global_step = 0
        wandb.init(project='SIFA', resume='allow', anonymous='must')
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    images = images.to(device=self.device, dtype=torch.float32,
                                       memory_format=torch.channels_last)  # todo float check
                    true_masks = true_masks.to(
                        device=self.device, dtype=torch.long)
                    true_labels = true_labels.to(
                        device=self.device, dtype=torch.float32)
                    outputs = model(batch)

                    prob_real_a_is_real = outputs['prob_real_a_is_real']
                    prob_real_b_is_real = outputs['prob_real_b_is_real']
                    fake_images_a = outputs['fake_images_a']
                    fake_images_b = outputs['fake_images_b']
                    prob_fake_a_is_real = outputs['prob_fake_a_is_real']
                    prob_fake_b_is_real = outputs['prob_fake_b_is_real']

                    cycle_images_a = outputs['cycle_images_a']
                    cycle_images_b = outputs['cycle_images_b']
                    prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
                    prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
                    pred_mask_a = outputs['pred_mask_a']
                    pred_mask_b = outputs['pred_mask_b']
                    pred_mask_b_ll = outputs['pred_mask_b_ll']
                    pred_mask_fake_a = outputs['pred_mask_fake_a']
                    pred_mask_fake_b = outputs['pred_mask_fake_b']
                    pred_mask_fake_b_ll = outputs['pred_mask_fake_b_ll']
                    prob_pred_mask_fake_b_is_real = outputs['prob_pred_mask_fake_b_is_real']
                    prob_pred_mask_b_is_real = outputs['prob_pred_mask_b_is_real']
                    prob_pred_mask_fake_b_ll_is_real = outputs['prob_pred_mask_fake_b_ll_is_real']
                    prob_pred_mask_b_ll_is_real = outputs['prob_pred_mask_b_ll_is_real']
                    prob_fake_a_aux_is_real = outputs['prob_fake_a_aux_is_real']
                    prob_fake_pool_a_aux_is_real = outputs['prob_fake_pool_a_aux_is_real']
                    prob_cycle_a_aux_is_real = outputs['prob_cycle_a_aux_is_real']

                    predicter_fake_b = torch.softmax(
                        pred_mask_fake_b, dim=3)
                    compact_pred_fake_b = torch.argmax(
                        predicter_fake_b, dim=3)
                    compact_y_fake_b = torch.argmax(gt_a, dim=3)
                    predicter_b = torch.softmax(pred_mask_b, dim=3)
                    compact_pred_b = torch.argmax(predicter_b, dim=3)
                    compact_y_b = torch.argmax(gt_b, dim=3)

                    dice_fake_b_arr = dice_eval(
                        compact_pred_fake_b, gt_a, self._num_cls)
                    dice_fake_b_mean = torch.mean(self.dice_fake_b_arr)
                    # For tensorboardX equivalent of tf.summary.scalar

                    dice_b_arr = dice_eval(
                        compact_pred_b, gt_b, self._num_cls)
                    dice_b_mean = torch.mean(dice_b_arr)

                    # For tensorboardX equivalent of tf.summary.scalar
                    g_loss_A, g_loss_B, seg_loss_B, d_loss_A, d_loss_B, d_loss_P, d_loss_P_ll = self.compute_losses(
                        input_a,
                        input_b,
                        cycle_images_a,
                        cycle_images_b, 
                        prob_fake_a_is_real,
                        prob_fake_b_is_real,
                        prob_pred_mask_b_is_real, 
                        prob_pred_mask_b_ll_is_real,
                        prob_fake_a_aux_is_real,
                        pred_mask_fake_b, 
                        pred_mask_fake_b_ll,
                        gt_a,
                        prob_real_a_is_real,
                        prob_fake_pool_a_is_real,
                        prob_cycle_a_aux_is_real, 
                        prob_fake_pool_a_aux_is_real, 
                        prob_real_b_is_real,
                        prob_fake_pool_b_is_real, 
                        prob_pred_mask_fake_b_is_real,
                        prob_pred_mask_b_is_real, 
                        prob_pred_mask_fake_b_ll_is_real, 
                        self._lambda_a, 
                        self._lambda_b
                    )
                    # Define the optimizers
                    # Zero the gradients, do backpropagation and update weights for each optimizer
                    def step_optimizer(optimizer, loss):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    step_optimizer(discriminator_A_optimizer, d_loss_A)
                    step_optimizer(discriminator_B_optimizer, d_loss_B)
                    step_optimizer(generator_A_optimizer, g_loss_A)
                    step_optimizer(generator_B_optimizer, g_loss_B)
                    step_optimizer(discriminator_P_optimizer, d_loss_P)
                    step_optimizer(discriminator_P_ll_optimizer, d_loss_P_ll)
                    step_optimizer(segmenter_B_optimizer, seg_loss_B)
                    


if __name__ == '__main__':

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
    main(config_filename=os_support_path('cyclegan/config_param.json'))
