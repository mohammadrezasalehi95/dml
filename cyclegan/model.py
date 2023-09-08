from cyclegan.layers import GeneralConv2D, GeneralDeconv2D
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# Load config file
with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])
POOL_SIZE = int(config['pool_size'])

# The height of each image.
IMG_HEIGHT = 256
# The width of each image.
IMG_WIDTH = 256
ngf = 32
ndf = 64


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=9):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_blocks = n_blocks
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        self.input_nc = input_nc
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class DiscriminatorAux(nn.Module):
    def __init__(self, ndf, f=4, padw=2):
        super(DiscriminatorAux, self).__init__()
        self.ndf = ndf
        self.f = f
        self.padw = padw
        self.conv1 = GeneralConv2D(1, ndf, f, 2, 0, activation_factor=0.2)
        self.conv2 = GeneralConv2D(ndf, ndf*2, f, 2, 0, activation_factor=0.2)
        self.conv3 = GeneralConv2D(ndf*2, ndf*4, f, 2, 0, activation_factor=0.2)
        self.conv4 = GeneralConv2D(ndf*4, ndf*8, f, 1, 0, activation_factor=0.2)
        self.conv5 = GeneralConv2D(ndf*8, 2, f, 1, 0, activation_factor=0, norm_type=None)

    def forward(self, x):
        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw))
        x = self.conv1(x)
        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw))
        x = self.conv2(x)
        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw))
        x = self.conv3(x)
        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw))
        x = self.conv4(x)
        x = F.pad(x, (self.padw, self.padw, self.padw, self.padw))
        x = self.conv5(x)
        
        return x[..., 0].unsqueeze(1), x[..., 1].unsqueeze(1)

class GeneratorResnet9blocks(nn.Module):
    def __init__(self, ngf=64, output_nc=1, skip=False):
        super(GeneratorResnet9blocks, self).__init__()

        # Define the reflection padding
        self.pad = nn.ReflectionPad2d(3)

        # Define the downsample convolutions
        self.conv1 = general_conv2d_ga(3, ngf, 7, 1, padding=0)
        self.conv2 = general_conv2d_ga(ngf, ngf * 2, 3, 2, padding=1)
        self.conv3 = general_conv2d_ga(ngf * 2, ngf * 4, 3, 2, padding=1)
        
        # Define the ResNet blocks
        self.resnet_blocks = nn.Sequential(
            *[build_resnet_block_ins(ngf * 4, padding=1) for _ in range(9)]
        )
        
        # Define the upsample convolutions
        self.deconv1 = general_deconv2d(ngf * 4, ngf * 2, 3, 2, padding=1, output_padding=1)
        self.deconv2 = general_deconv2d(ngf * 2, ngf, 3, 2, padding=1, output_padding=1)
        self.conv4 = general_conv2d_ga(ngf, output_nc, 7, 1, padding=0, norm=False, activation=False)

        self.skip = skip

    def forward(self, inputgen, inputimg):
        # Apply padding
        x = self.pad(inputgen)

        # Apply downsample convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Apply ResNet blocks
        x = self.resnet_blocks(x)

        # Apply upsample convolutions
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv4(x)

        # Apply skip connection if required
        if self.skip:
            return torch.tanh(inputimg + x)
        else:
            return torch.tanh(x)
# Then you can define your models like this:

class GeneratorResnet9blocks(nn.Module):
    def __init__(self, ngf=64, output_nc=1, skip=False):
        super(GeneratorResnet9blocks, self).__init__()

        # Define the reflection padding
        self.pad = nn.ReflectionPad2d(3)

        # Define the downsample convolutions
        self.conv1 = GeneralConv2D_GA(3, ngf, 7, stride=1, padding="valid", do_norm=True, do_relu=True, norm_type='Ins')
        self.conv2 = GeneralConv2D_GA(ngf, ngf * 2, 3, stride=2, padding="same", do_norm=True, do_relu=True, norm_type='Ins')
        self.conv3 = GeneralConv2D_GA(ngf * 2, ngf * 4, 3, stride=2, padding="same", do_norm=True, do_relu=True, norm_type='Ins')
        
        # Define the ResNet blocks
        self.resnet_blocks = nn.Sequential(
            *[ResnetBlock(ngf * 4) for _ in range(9)]
        )
        
        # Define the upsample convolutions
        self.deconv1 = GeneralDeconv2D(ngf * 4, ngf * 2, 3, stride=2, padding="same", do_norm=True, do_relu=True, norm_type='Ins')
        self.deconv2 = GeneralDeconv2D(ngf * 2, ngf, 3, stride=2, padding="same", do_norm=True, do_relu=True, norm_type='Ins')
        self.conv4 = GeneralConv2D_GA(ngf, output_nc, 7, stride=1, padding="valid", do_norm=False, do_relu=False)
        
        self.skip = skip

    def forward(self, inputgen, inputimg):
        # Apply padding
        x = self.pad(inputgen)

        # Apply downsample convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Apply ResNet blocks
        x = self.resnet_blocks(x)

        # Apply upsample convolutions
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv4(x)

        # Apply skip connection if required
        if self.skip:
            return torch.tanh(inputimg + x)
        else:
            return torch.tanh(x)





class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,skip
                 , bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.discriminator_aux_A = DiscriminatorAux()
        self.discriminator_aux_B = DiscriminatorAux()
        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()
        self.discriminator_P = Discriminator()
        self.discriminator_P_ll = Discriminator()
        self.generator_A = Generator(n_blocks=9)
        self.encoder_B = Encoder()
        self.decoder_B = Decoder()
        self.segmenter_B = Segmenter()
        self.segmenter_B_ll = Segmenter()
        self.drop_out_rate=0.25
        self.fc=nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, inputs,skip=False):
        images_a = inputs['images_a']
        images_b = inputs['images_b']
        fake_pool_a = inputs['fake_pool_a']
        fake_pool_b = inputs['fake_pool_b']
        
        prob_real_a_is_real, prob_real_a_aux = self.discriminator_aux_A(images_a)
        prob_real_b_is_real = self.discriminator_B(images_b)

        fake_images_b = self.generator_A(images_a, images_a, skip=skip)
        latent_b, latent_b_ll = self.encoder_B(images_b, skip=skip, drop_out_rate=self.drop_out_rate)
        fake_images_a = self.decoder_B(latent_b, images_b, skip=skip)
        pred_mask_b = self.segmenter_B(latent_b)
        pred_mask_b_ll = self.segmenter_B_ll(latent_b_ll)
        prob_fake_a_is_real, prob_fake_a_aux_is_real = self.discriminator_aux_A(fake_images_a)
        prob_fake_b_is_real = self.discriminator_B(fake_images_b)

        latent_fake_b, latent_fake_b_ll = self.encoder_B(fake_images_b, skip=skip, drop_out_rate=self.drop_out_rate)
        cycle_images_b = self.generator_A(fake_images_a, fake_images_a, skip=skip)

        cycle_images_a = self.decoder_B(latent_fake_b, fake_images_b, skip=skip)

        pred_mask_fake_b = self.segmenter_B(latent_fake_b,drop_out_rate=self.drop_out_rate)
        pred_mask_fake_b_ll = self.segmenter_B_ll(latent_fake_b_ll,drop_out_rate=self.drop_out_rate)

        prob_fake_pool_a_is_real, prob_fake_pool_a_aux_is_real = self.discriminator_aux_A(fake_pool_a)
        prob_fake_pool_b_is_real = self.discriminator_B(fake_pool_b)

        # prob_fake_pool_a_is_real, prob_fake_pool_a_aux_is_real = discriminator_aux(fake_pool_a, "d_A")
        # prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, "d_B")

        # prob_cycle_a_is_real, prob_cycle_a_aux_is_real = discriminator_aux(cycle_images_a, "d_A")

        # prob_pred_mask_fake_b_is_real = current_discriminator(pred_mask_fake_b, name="d_P")
        # prob_pred_mask_b_is_real = current_discriminator(pred_mask_b, 'd_P')

        # prob_pred_mask_fake_b_ll_is_real = current_discriminator(pred_mask_fake_b_ll, name="d_P_ll")
        # prob_pred_mask_b_ll_is_real = current_discriminator(pred_mask_b_ll, 'd_P_ll')

        return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
        'pred_mask_a': pred_mask_b,
        'pred_mask_b': pred_mask_b,
        'pred_mask_b_ll': pred_mask_b_ll,
        'pred_mask_fake_a': pred_mask_fake_b,
        'pred_mask_fake_b': pred_mask_fake_b,
        'pred_mask_fake_b_ll': pred_mask_fake_b_ll,
        'prob_pred_mask_fake_b_is_real': prob_pred_mask_fake_b_is_real,
        'prob_pred_mask_b_is_real': prob_pred_mask_b_is_real,
        'prob_pred_mask_fake_b_ll_is_real': prob_pred_mask_fake_b_ll_is_real,
        'prob_pred_mask_b_ll_is_real': prob_pred_mask_b_ll_is_real,
        'prob_fake_a_aux_is_real': prob_fake_a_aux_is_real,
        'prob_fake_pool_a_aux_is_real': prob_fake_pool_a_aux_is_real,
        'prob_cycle_a_aux_is_real': prob_cycle_a_aux_is_real,
    }

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



