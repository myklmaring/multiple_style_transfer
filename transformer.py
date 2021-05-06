import torch.nn as nn
from torch.nn import functional


class Transformer(nn.Module):
    def __init__(self, padding='reflect'):
        super(Transformer, self).__init__()

        # Relu layer after all normalization, no ReLU after Residual Layers
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1, padding=padding)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, padding=padding)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2, padding=padding)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)

        self.encoder = nn.Sequential(
            self.conv1,
            self.norm1,
            self.relu,
            self.conv2,
            self.norm2,
            self.relu,
            self.conv3,
            self.norm3,
            self.relu
        )

        # Residual
        self.res1 = ResidualLayer(128, padding=padding)
        self.res2 = ResidualLayer(128, padding=padding)
        self.res3 = ResidualLayer(128, padding=padding)
        self.res4 = ResidualLayer(128, padding=padding)
        self.res5 = ResidualLayer(128, padding=padding)

        self.residual = nn.Sequential(
            self.res1,
            self.res2,
            self.res3,
            self.res4,
            self.res5
        )

        # Decoder
        self.up1 = Upsample(128, 64, kernel_size=3, stride=1, factor=2, padding=padding)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
        self.up2 = Upsample(64, 32, kernel_size=3, stride=1, factor=2, padding=padding)
        self.norm5 = nn.InstanceNorm2d(32, affine=True)
        self.conv4 = ConvLayer(32, 3, kernel_size=9, stride=1, padding=padding)

        self.decoder = nn.Sequential(
            self.up1,
            self.norm4,
            self.relu,
            self.up2,
            self.norm5,
            self.relu,
            self.conv4
        )

    def forward(self, x):
        output = self.encoder(x)
        output = self.residual(output)
        output = self.decoder(output)

        return output


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding='reflect'):
        super(ConvLayer, self).__init__()

        # pad to maintain output H, W
        pad = (kernel_size - 1) // 2
        self.pad1 = Padding(pad, padding)

        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride)

    def forward(self, x):
        output = self.pad1(x)
        output = self.conv1(output)
        return output


class Upsample(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, factor, padding='reflect'):
        super(Upsample, self).__init__()

        pad = (kernel_size-1) // 2
        self.up1 = Interpolate(factor=factor, mode='nearest')
        self.pad1 = Padding(pad, padding)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        output = self.up1(x)
        output = self.pad1(output)
        output = self.conv1(output)
        return output


class ResidualLayer(nn.Module):
    def __init__(self, dim, padding='reflect'):
        super(ResidualLayer, self).__init__()

        self.ConvLayer1 = ConvLayer(dim, dim, kernel_size=3, stride=1, padding=padding)
        self.norm1 = nn.InstanceNorm2d(dim, affine=True)
        self.ConvLayer2 = ConvLayer(dim, dim, kernel_size=3, stride=1, padding=padding)
        self.norm2 = nn.InstanceNorm2d(dim, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        output = self.ConvLayer1(x)
        output = self.norm1(output)
        output = self.relu(output)
        output = self.ConvLayer2(output)
        output = self.norm2(output)
        output += residual
        return output


class Interpolate(nn.Module):
    def __init__(self, factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.factor = factor
        self.mode = mode

    def forward(self, x):
        output = self.interp(x, scale_factor=self.factor, mode=self.mode)
        return output

def Padding(pad, padding):
    if padding == 'reflect':
        return nn.ReflectionPad2d(pad)
    elif padding == 'replicate':
        return nn.ReplicationPad2d(pad)
