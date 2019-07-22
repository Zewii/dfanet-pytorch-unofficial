#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.nn.modules import Module
from torch.nn.modules import Conv2d, MaxPool2d, BatchNorm2d, PReLU, ReLU, Sequential, Dropout2d, ConvTranspose2d, MaxUnpool2d
# from torchsummary import summary


class InitialBlock(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation='PReLU',
                 bias=False):
        super(InitialBlock, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1)
        self.maxpooling = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bnActivate = Sequential(
            BatchNorm2d(out_channels),
            PReLU() if activation == 'PReLU' else ReLU())

    def forward(self, x):
        init_conv = self.conv(x)
        init_maxpool = self.maxpooling(x)
        init_concat = torch.cat((init_conv, init_maxpool), 1)
        out = self.bnActivate(init_concat)
        return out


class RegularBottleNeck(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 stride=1,
                 padding=1,
                 activation='PReLU',
                 bias=False,
                 asymmetric=False,
                 dropout_prob=0):
        super(RegularBottleNeck, self).__init__()
        internal_channels = in_channels // 4
        self.conv_down = Sequential(
            Conv2d(
                in_channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias), BatchNorm2d(internal_channels),
            PReLU() if activation == 'PReLU' else ReLU())
        if asymmetric is False:
            self.conv_main = Sequential(
                Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    padding=padding,
                    bias=bias), BatchNorm2d(internal_channels),
                PReLU() if activation == 'PReLU' else ReLU())
        else:
            self.conv_main = Sequential(
                Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    dilation=dilation,
                    stride=stride,
                    padding=(padding, 0),
                    bias=bias), BatchNorm2d(internal_channels),
                PReLU() if activation == 'PReLU' else ReLU(),
                Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    dilation=dilation,
                    stride=stride,
                    padding=(0, padding),
                    bias=bias), BatchNorm2d(internal_channels),
                PReLU() if activation == 'PReLU' else ReLU())
        self.conv_up = Sequential(
            Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias), BatchNorm2d(out_channels),
            PReLU() if activation == 'PReLU' else ReLU())
        self.regularizer = Dropout2d(p=dropout_prob)
        self.out_activation = PReLU() if activation == 'PReLU' else ReLU()

    def forward(self, x):
        main = x
        ext = self.conv_down(x)
        ext = self.conv_main(ext)
        ext = self.conv_up(ext)
        ext = self.regularizer(ext)
        out = main + ext
        return self.out_activation(out)


class DownsamplingBottleNeck(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 stride=1,
                 padding=1,
                 activation='PReLU',
                 bias=False,
                 asymmetric=False,
                 dropout_prob=0,
                 return_indices=True):
        super(DownsamplingBottleNeck, self).__init__()
        internal_channels = in_channels // 4
        self.conv_down = Sequential(
            Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias), BatchNorm2d(internal_channels),
            PReLU() if activation == 'PReLU' else ReLU())
        if asymmetric is False:
            self.conv_main = Sequential(
                Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    padding=padding,
                    bias=bias), BatchNorm2d(internal_channels),
                PReLU() if activation == 'PReLU' else ReLU())
        else:
            self.conv_main = Sequential(
                Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    dilation=dilation,
                    stride=stride,
                    padding=(padding, 0),
                    bias=bias), BatchNorm2d(internal_channels),
                PReLU() if activation == 'PReLU' else ReLU(),
                Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    dilation=dilation,
                    stride=stride,
                    padding=(0, padding),
                    bias=bias), BatchNorm2d(internal_channels),
                PReLU() if activation == 'PReLU' else ReLU())
        self.conv_up = Sequential(
            Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias), BatchNorm2d(out_channels),
            PReLU() if activation == 'PReLU' else ReLU())
        self.regularizer = Dropout2d(p=dropout_prob)
        self.out_activation = PReLU() if activation == 'PReLU' else ReLU()
        self.main_branch = MaxPool2d(
            kernel_size=2, stride=2, padding=0, return_indices=return_indices)

    def forward(self, x):
        main, indices = self.main_branch(x)
        ext = self.conv_down(x)
        ext = self.conv_main(ext)
        ext = self.conv_up(ext)
        ext = self.regularizer(ext)
        main = torch.cat(
            (main,
             torch.zeros(main.shape[0], ext.shape[1] - main.shape[1],
                         *main.shape[2:4]).cpu()),
            dim=1)
        out = main + ext
        return self.out_activation(out), indices


class UpsamplingBottleNeck(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 stride=2,
                 padding=1,
                 output_padding=1,
                 activation='PReLU',
                 bias=False,
                 dropout_prob=0.1):
        super(UpsamplingBottleNeck, self).__init__()
        internal_channels = in_channels // 4
        self.conv_down = Sequential(
            Conv2d(
                in_channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias), BatchNorm2d(internal_channels),
            PReLU() if activation == 'PReLU' else ReLU())
        self.conv_main = Sequential(
            ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                bias=bias), BatchNorm2d(internal_channels),
            PReLU() if activation == 'PReLU' else ReLU())
        self.conv_up = Sequential(
            Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias), BatchNorm2d(out_channels),
            PReLU() if activation == 'PReLU' else ReLU())
        self.main_conv = Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias), BatchNorm2d(out_channels))
        self.mainmaxunpool = MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.regularizer = Dropout2d(p=dropout_prob)
        self.out_activation = PReLU() if activation == 'PReLU' else ReLU()

    def forward(self, x, indices):
        main = self.main_conv(x)
        main = self.mainmaxunpool(main, indices)
        ext = self.conv_down(x)
        ext = self.conv_main(ext)
        ext = self.conv_up(ext)
        ext = self.regularizer(ext)
        out = main + ext
        return self.out_activation(out)


class ENet(Module):
    def __init__(self, nb_classes):
        super(ENet, self).__init__()
        self.init_block = InitialBlock(3, 16)
        self.downsampling1 = DownsamplingBottleNeck(
            in_channels=16,
            out_channels=64,
            kernel_size=3,
            dilation=1,
            stride=1,
            padding=1,
            dropout_prob=0.01)
        self.first_stage = Sequential(
            RegularBottleNeck(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                dropout_prob=0.01),
            RegularBottleNeck(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                dropout_prob=0.01),
            RegularBottleNeck(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                dropout_prob=0.01),
            RegularBottleNeck(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                dropout_prob=0.01))
        self.downsampling2 = DownsamplingBottleNeck(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            dilation=1,
            stride=1,
            padding=1,
            dropout_prob=0.1)
        self.second_stage = Sequential(
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                dilation=1,
                stride=1,
                padding=2,
                asymmetric=True,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=4,
                stride=1,
                padding=4,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=8,
                stride=1,
                padding=8,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                dilation=1,
                stride=1,
                padding=2,
                asymmetric=True,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=16,
                stride=1,
                padding=16,
                dropout_prob=0.1))
        self.third_stage = Sequential(
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=2,
                stride=1,
                padding=2,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                dilation=1,
                stride=1,
                padding=2,
                asymmetric=True,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=4,
                stride=1,
                padding=4,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=8,
                stride=1,
                padding=8,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                dilation=1,
                stride=1,
                padding=2,
                asymmetric=True,
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dilation=16,
                stride=1,
                padding=16,
                dropout_prob=0.1))
        self.upsampling1 = UpsamplingBottleNeck(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            dilation=1,
            stride=2,
            padding=1,
            output_padding=1,
            activation='PReLU',
            dropout_prob=0.1)
        self.forth_stage = Sequential(
            RegularBottleNeck(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                activation='PReLU',
                dropout_prob=0.1),
            RegularBottleNeck(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                dilation=1,
                stride=1,
                padding=1,
                activation='PReLU',
                dropout_prob=0.1))
        self.upsampling2 = UpsamplingBottleNeck(
            in_channels=64,
            out_channels=16,
            kernel_size=3,
            dilation=1,
            stride=2,
            padding=1,
            output_padding=1,
            activation='PReLU',
            dropout_prob=0.1)
        self.fifth_stage = RegularBottleNeck(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            dilation=1,
            stride=1,
            padding=1,
            activation='PReLU',
            dropout_prob=0.1)
        self.fullconv = ConvTranspose2d(
            in_channels=16,
            out_channels=nb_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

    def forward(self, x):
        x = self.init_block(x)
        x, indices1 = self.downsampling1(x)
        x = self.first_stage(x)
        x, indices2 = self.downsampling2(x)
        x = self.second_stage(x)
        x = self.third_stage(x)
        x = self.upsampling1(x, indices2)
        x = self.forth_stage(x)
        x = self.upsampling2(x, indices1)
        x = self.fifth_stage(x)
        x = self.fullconv(x)
        return x


if __name__ == '__main__':
    model = ENet(nb_classes=13)
    model.cuda()
    model.eval()
    example = torch.rand(2, 3, 1024, 1024).cuda()
    summary(model, (3, 1024, 1024))
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("../res/model_no_shared512.pt")
