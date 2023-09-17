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


class GeneralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=7, stride=1, padding=0, stddev=0.01, do_norm=True, do_relu=True, p_drop=None, relufactor=0, norm_type=None):
        super(GeneralConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.normal_(self.conv.weight, std=stddev)
        self.do_norm = do_norm
        self.norm_type = norm_type
        if self.do_norm:
            if self.norm_type == 'Batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif self.norm_type == 'Ins':
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise ValueError("Normalization type is not specified or not recognized!")
        self.do_relu = do_relu
        self.relufactor = relufactor
        self.dropout = nn.Dropout(p_drop) if p_drop is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.do_norm:
            x = self.norm(x)
        if self.do_relu:
            if self.relufactor == 0:
                x = F.relu(x)
            else:
                x = F.leaky_relu(x, negative_slope=self.relufactor)
        return x

class GeneralConv2dGA(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=7, stride=1, padding=0, stddev=0.02, do_norm=True, do_relu=True, p_drop=None, relufactor=0, norm_type=None):
        super(GeneralConv2dGA, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.normal_(self.conv.weight, std=stddev)
        nn.init.constant_(self.conv.bias, 0)  # biases are initialized to 0
        self.do_norm = do_norm
        self.norm_type = norm_type
        if self.do_norm:
            if self.norm_type == 'Batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif self.norm_type == 'Ins':
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise ValueError("Normalization type is not specified or not recognized!")
        self.do_relu = do_relu
        self.relufactor = relufactor
        self.dropout = nn.Dropout(p_drop) if p_drop is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.do_norm:
            x = self.norm(x)
        if self.do_relu:
            if self.relufactor == 0:
                x = F.relu(x)
            else:
                x = F.leaky_relu(x, negative_slope=self.relufactor)
        return x



class DilateConv2d(GeneralConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=2,
                 padding="valid", do_norm=True, do_relu=True, relu_factor=0,
                 norm_type=None):
        super(DilateConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, do_norm=do_norm, do_relu=do_relu, relu_factor=relu_factor, norm_type=norm_type)
        if padding.lower() == "valid":
            padding = 0
        elif padding.lower() == "same":
            padding = dilation * (kernel_size // 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation)


    
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
    
class GeneralDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1,
                 padding="valid", do_norm=True, do_relu=True, relu_factor=0,
                 norm_type=None):
        super(GeneralDeconv2d, self).__init__()
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
