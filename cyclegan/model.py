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


# Then you can define your models like this:
netG_A = Generator(input_nc, output_nc)
netG_B = Generator(input_nc, output_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(input_nc)