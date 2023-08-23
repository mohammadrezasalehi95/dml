import torch
import torch.nn as nn
import torch.nn.functional as F


def lrelu(x, leak=0.2, alt_relu_impl=False):
    if alt_relu_impl:
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * torch.abs(x)
    else:
        return F.leaky_relu(x, negative_slope=leak)


class InstanceNorm(nn.Module):
    def __init__(self, num_features):
        super(InstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features)

    def forward(self, x):
        return self.norm(x)


class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.norm(x)


class GeneralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1,
                 padding="valid", do_norm=True, do_relu=True, relu_factor=0,
                 norm_type=None):
        super(GeneralConv2D, self).__init__()
        if padding.lower() == "valid":
            padding = 0
        elif padding.lower() == "same":
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.do_norm = do_norm
        self.do_relu = do_relu
        self.relu_factor = relu_factor

        if self.do_norm:
            if norm_type is None:
                raise ValueError("normalization type is not specified!")
            elif norm_type.lower() == 'ins':
                self.norm = InstanceNorm(out_channels)
            elif norm_type.lower() == 'batch':
                self.norm = BatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)

        if self.do_norm:
            x = self.norm(x)

        if self.do_relu:
            if self.relu_factor == 0:
                x = F.relu(x)
            else:
                x = lrelu(x, self.relu_factor)

        return x



class DilateConv2D(GeneralConv2D):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=2,
                 padding="valid", do_norm=True, do_relu=True, relu_factor=0,
                 norm_type=None):
        super(DilateConv2D, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, do_norm=do_norm, do_relu=do_relu, relu_factor=relu_factor, norm_type=norm_type)
        if padding.lower() == "valid":
            padding = 0
        elif padding.lower() == "same":
            padding = dilation * (kernel_size // 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation)


class GeneralDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1,
                 padding="valid", do_norm=True, do_relu=True, relu_factor=0,
                 norm_type=None):
        super(GeneralDeconv2D, self).__init__()
        if padding.lower() == "valid":
            padding = 0
        elif padding.lower() == "same":
            padding = kernel_size // 2

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.do_norm = do_norm
        self.do_relu = do_relu
        self.relu_factor = relu_factor

        if self.do_norm:
            if norm_type is None:
                raise ValueError("normalization type is not specified!")
            elif norm_type.lower() == 'ins':
                self.norm = InstanceNorm(out_channels)
            elif norm_type.lower() == 'batch':
                self.norm = BatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)

        if self.do_norm:
            x = self.norm(x)

        if self.do_relu:
            if self.relu_factor == 0:
                x = F.relu(x)
            else:
                x = lrelu(x, self.relu_factor)

        return x
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class LRelu(nn.Module):
    def __init__(self, leak=0.2, alt_relu_impl=False):
        super(LRelu, self).__init__()
        self.leak = leak
        self.alt_relu_impl = alt_relu_impl

    def forward(self, x):
        if self.alt_relu_impl:
            f1 = 0.5 * (1 + self.leak)
            f2 = 0.5 * (1 - self.leak)
            return f1 * x + f2 * torch.abs(x)
        else:
            return F.leaky_relu(x, negative_slope=self.leak)


class GeneralConv2D(nn.Module):
    def __init__(self, inputconv, o_d=64, f=7, s=1, stddev=0.01,
                 padding="VALID", name="conv2d", do_norm=True, do_relu=True, 
                 relufactor=0, norm_type=None, is_training=True):
        super(GeneralConv2D, self).__init__()

        self.conv = nn.Conv2d(inputconv, o_d, f, stride=s, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(o_d) if norm_type == 'Batch' else nn.InstanceNorm2d(o_d)
        self.do_norm = do_norm
        self.do_relu = do_relu
        self.lrelu = LRelu(relufactor)
        self.is_training = is_training

    def forward(self, x):
        x = self.conv(x)
        if self.do_norm:
            x = self.norm(x)
        if self.do_relu:
            x = self.lrelu(x)
        return x


class GeneralDeconv2D(nn.Module):
    def __init__(self, inputconv, outshape, o_d=64, f=7, s=1,
                 stddev=0.02, padding="VALID", name="deconv2d",
                 do_norm=True, do_relu=True, relufactor=0, norm_type=None, is_training=True):
        super(GeneralDeconv2D, self).__init__()

        self.conv_t = nn.ConvTranspose2d(inputconv, o_d, f, stride=s, padding=1, output_padding=1, bias=False)
        self.norm = nn.BatchNorm2d(o_d) if norm_type == 'Batch' else nn.InstanceNorm2d(o_d)
        self.do_norm = do_norm
        self.do_relu = do_relu
        self.lrelu = LRelu(relufactor)
        self.is_training = is_training

    def forward(self, x):
        x = self.conv_t(x)
        if self.do_norm:
            x = self.norm(x)
        if self.do_relu:
            x = self.lrelu(x)
        return x