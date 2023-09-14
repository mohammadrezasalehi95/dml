from layers import *
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
    def __init__(self, dim, padding_type='reflect', norm_type=None, use_dropout=True, dropout_rate=0.25):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_type, use_dropout, dropout_rate)

    def build_conv_block(self, dim, padding_type, norm_type, use_dropout, dropout_rate):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        conv_block += [GeneralConv2d(dim, dim, kernel_size=3, stride=1, padding=p),
                       nn.ReLU(True)]
        
        if use_dropout:
            conv_block += [nn.Dropout(dropout_rate)]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        conv_block += [GeneralConv2d(dim, dim, kernel_size=3, stride=1, padding=p)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out_res = x + self.conv_block(x)
        return out_res


class ResnetBlockIns(ResnetBlock):
    def build_conv_block(self, dim, padding_type, norm_type, use_dropout, dropout_rate):
        conv_block = super().build_conv_block(dim, padding_type, norm_type='Ins', 
                                            #   use_dropout, 
                                            #   dropout_rate
                                              )
        return conv_block


class ResnetBlockDS(ResnetBlock):
    def __init__(self, dim_in, dim_out, padding_type='reflect', norm_type=None, use_dropout=True, dropout_rate=0.25):
        super(ResnetBlockDS, self).__init__(dim_out, padding_type, norm_type, use_dropout, dropout_rate)
        self.expand_dim = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0) if dim_in != dim_out else None

    def forward(self, x):
        if self.expand_dim is not None:
            x = self.expand_dim(x)
        out_res = x + self.conv_block(x)
        return out_res

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
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(DiscriminatorAux, self).__init__()

        self.padding = nn.ConstantPad2d(2, 0)

        self.conv1 = GeneralConv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, norm_layer=norm_layer, use_bias=False, activation=nn.LeakyReLU(0.2, True))
        self.conv2 = GeneralConv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, norm_layer=norm_layer, use_bias=False, activation=nn.LeakyReLU(0.2, True))
        self.conv3 = GeneralConv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=0, norm_layer=norm_layer, use_bias=False, activation=nn.LeakyReLU(0.2, True))
        self.conv4 = GeneralConv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=0, norm_layer=norm_layer, use_bias=False, activation=nn.LeakyReLU(0.2, True))
        self.conv5 = GeneralConv2d(ndf * 8, 2, kernel_size=4, stride=1, padding=0, use_bias=False, activation=None)

    def forward(self, input):
        x = self.padding(input)
        x = self.conv1(x)
        x = self.padding(x)
        x = self.conv2(x)
        x = self.padding(x)
        x = self.conv3(x)
        x = self.padding(x)
        x = self.conv4(x)
        x = self.padding(x)
        x = self.conv5(x)
        
        return x[..., 0].unsqueeze(1), x[..., 1].unsqueeze(1)
    

class DRNBlock(nn.Module):
    def __init__(self, dim, norm_layer=None, dropout_rate=0.25):
        super(DRNBlock, self).__init__()
        
        padding = 2
        dilation = 2

        self.pad1 = nn.ReflectionPad2d(padding)
        self.dilated_conv1 = DilateConv2d(dim, dim, kernel_size=3, stride=1, dilation=dilation, bias=False, norm_layer=norm_layer, activation=nn.ReLU(True), dropout_rate=dropout_rate)
        
        self.pad2 = nn.ReflectionPad2d(padding)
        self.dilated_conv2 = DilateConv2d(dim, dim, kernel_size=3, stride=1, dilation=dilation, bias=False, norm_layer=norm_layer, activation=None, dropout_rate=dropout_rate)

    def forward(self, input):
        x = self.pad1(input)
        x = self.dilated_conv1(x)
        x = self.pad2(x)
        x = self.dilated_conv2(x)

        return F.relu(x + input)
    
class DRNBlockDS(nn.Module):
    def __init__(self, dim_in, dim_out, norm_layer=None, dropout_rate=0.25):
        super(DRNBlockDS, self).__init__()
        
        padding = 2
        dilation = 2

        self.pad1 = nn.ReflectionPad2d(padding)
        self.dilated_conv1 = DilateConv2d(dim_in, dim_out, kernel_size=3, stride=1, dilation=dilation, bias=False, norm_layer=norm_layer, activation=nn.ReLU(True), dropout_rate=dropout_rate)
        
        self.pad2 = nn.ReflectionPad2d(padding)
        self.dilated_conv2 = DilateConv2d(dim_out, dim_out, kernel_size=3, stride=1, dilation=dilation, bias=False, norm_layer=norm_layer, activation=None, dropout_rate=dropout_rate)

        self.pad3 = nn.ReflectionPad2d((dim_out-dim_in)//2)

    def forward(self, input):
        x = self.pad1(input)
        x = self.dilated_conv1(x)
        x = self.pad2(x)
        x = self.dilated_conv2(x)

        input = self.pad3(input)

        return F.relu(x + input)
class DiscriminatorAux(nn.Module):
    def __init__(self, ndf, f=4, padw=2):
        super(DiscriminatorAux, self).__init__()
        self.ndf = ndf
        self.f = f
        self.padw = padw
        self.conv1 = GeneralConv2d(1, ndf, f, 2, 0, activation_factor=0.2)
        self.conv2 = GeneralConv2d(ndf, ndf*2, f, 2, 0, activation_factor=0.2)
        self.conv3 = GeneralConv2d(ndf*2, ndf*4, f, 2, 0, activation_factor=0.2)
        self.conv4 = GeneralConv2d(ndf*4, ndf*8, f, 1, 0, activation_factor=0.2)
        self.conv5 = GeneralConv2d(ndf*8, 2, f, 1, 0, activation_factor=0, norm_type=None)

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

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc,output_nc=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect',skip=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.skip=skip
        model = []
        model += [nn.ReflectionPad2d(3),
                  GeneralConv2dGA(input_nc, ngf, kernel_size=7, padding=0)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [GeneralConv2dGA(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlockIns(ngf * mult, padding_type=padding_type, norm_type='instance', use_dropout=use_dropout, dropout_rate=0.25)]
   
        #upsample convolutions
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [GeneralDeconv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1)]


        model += [nn.ReflectionPad2d(3),
                  GeneralConv2dGA(ngf, output_nc, kernel_size=7, padding=0,do_relu=False),
                  ]
        self.model = nn.Sequential(*model)
        

    def forward(self,  input_gen, input_img):
        x=self.model(input_gen)
        if self.skip:
            x = torch.tanh(input_img + x)
        else:
            x = torch.tanh(x)

        return x



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

class Decoder(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_bias=False, skip=False):
        super(Decoder, self).__init__()

        ks = 3
        f = 7
        padding_type = 'reflect'
        
        self.skip = skip

        self.conv1 = GeneralConv2d(input_nc, ngf * 4, ks, stride=1, padding=ks//2, norm_layer=norm_layer, use_bias=use_bias, activation=nn.ReLU(True))
        self.res1 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias)
        self.res2 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias)
        self.res3 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias)
        self.res4 = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias)
        self.deconv1 = GeneralDeconv2d(ngf * 4, ngf * 2, ks, stride=2, padding=ks//2, output_padding=1, norm_layer=norm_layer, use_bias=use_bias, activation=nn.ReLU(True))
        self.deconv2 = GeneralDeconv2d(ngf * 2, ngf * 2, ks, stride=2, padding=ks//2, output_padding=1, norm_layer=norm_layer, use_bias=use_bias, activation=nn.ReLU(True))
        self.deconv3 = GeneralDeconv2d(ngf * 2, ngf, ks, stride=2, padding=ks//2, output_padding=1, norm_layer=norm_layer, use_bias=use_bias, activation=nn.ReLU(True))
        self.conv2 = GeneralConv2d(ngf, 1, f, stride=1, padding=f//2, norm_layer=None, use_bias=use_bias, activation=None)

    def forward(self, input, input_img):
        x = self.conv1(input)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.conv2(x)
        
        if self.skip:
            out_gen = torch.tanh(input_img + x)
        else:
            out_gen = torch.tanh(x)

        return out_gen

class Segmenter(nn.Module):
    def __init__(self, input_nc, keep_rate=0.75):
        super(Segmenter, self).__init__()

        self.conv = GeneralConv2d(input_nc, 5, kernel_size=1, stride=1, padding=0, norm_layer=None, use_bias=False, activation=None)
        self.dropout = nn.Dropout(p=1-keep_rate)
        
    def forward(self, input):
        x = self.conv(input)
        x = self.dropout(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, input_dim, name='encoder', is_training=True, keep_rate=0.75):
        super(Encoder, self).__init__()
        self.fb = 16
        self.k1 = 3
        self.padding = "constant"
        self.keep_rate = keep_rate
        self.is_training = is_training

        # Define layers
        self.c1 = GeneralConv2d(input_dim, self.fb, 7, 1, padding=0, stddev=0.01)
        self.r1 = ResnetBlock(self.fb, padding_type=self.padding)
        self.r2 = ResnetBlockDS(self.fb, self.fb*2, padding_type=self.padding)
        self.r3 = ResnetBlockDS(self.fb*2, self.fb*4, padding_type=self.padding)
        self.r4 = ResnetBlock(self.fb*4, padding_type=self.padding)
        self.r5 = ResnetBlockDS(self.fb*4, self.fb*8, padding_type=self.padding)
        self.r6 = ResnetBlock(self.fb*8, padding_type=self.padding)
        self.r7 = ResnetBlockDS(self.fb*8, self.fb*16, padding_type=self.padding)
        self.r8 = ResnetBlock(self.fb*16, padding_type=self.padding)
        self.r9 = ResnetBlock(self.fb*16, padding_type=self.padding)
        self.r10 = ResnetBlock(self.fb*16, padding_type=self.padding)
        self.r11 = ResnetBlockDS(self.fb*16, self.fb*32, padding_type=self.padding)
        self.r12 = ResnetBlock(self.fb*32, padding_type=self.padding)
        self.d1 = DRNBlock(self.fb*32)
        self.d2 = DRNBlock(self.fb*32)
        self.c2 = GeneralConv2d(self.fb*32, self.fb*32, self.k1, 1, stddev=0.01)
        self.c3 = GeneralConv2d(self.fb*32, self.fb*32, self.k1, 1, stddev=0.01)

    def forward(self, x):
        o_c1 = self.c1(x)
        o_r1 = self.r1(o_c1)
        out1 = F.max_pool2d(o_r1, 2, stride=2)

        o_r2 = self.r2(out1)
        out2 = F.max_pool2d(o_r2, 2, stride=2)

        o_r3 = self.r3(out2)
        o_r4 = self.r4(o_r3)
        out3 = F.max_pool2d(o_r4, 2, stride=2)

        o_r5 = self.r5(out3)
        o_r6 = self.r6(o_r5)

        o_r7 = self.r7(o_r6)
        o_r8 = self.r8(o_r7)

        o_r9 = self.r9(o_r8)
        o_r10 = self.r10(o_r9)

        o_r11 = self.r11(o_r10)
        o_r12 = self.r12(o_r11)

        o_d1 = self.d1(o_r12)
        o_d2 = self.d2(o_d1)

        o_c2 = self.c2(o_d2)
        o_c3 = self.c3(o_c2)

        return o_c3, o_r12
class SIFAModule(nn.Module):
    def __init__(self, n_channels, n_classes,skip
                 , bilinear=False):
        super(SIFAModule, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.discriminator_aux_A = DiscriminatorAux()
        self.discriminator_B = Discriminator()
        self.discriminator_P = Discriminator()
        self.discriminator_P_ll = Discriminator()
        self.generator_A = ResnetGenerator(n_blocks=9)
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
        prob_cycle_a_is_real, prob_cycle_a_aux_is_real = self.discriminator_aux_A(cycle_images_a)
        prob_pred_mask_fake_b_is_real = self.discriminator_P(pred_mask_fake_b)
        prob_pred_mask_b_is_real = self.discriminator_P(pred_mask_b)
        prob_pred_mask_fake_b_ll_is_real = self.discriminator_P_ll(pred_mask_fake_b_ll)
        prob_pred_mask_b_ll_is_real = self.discriminator_P_ll(pred_mask_b_ll)
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
