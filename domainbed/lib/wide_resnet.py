# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
From https://github.com/meliketoy/wide-resnet.pytorch
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

# 虽然PyTorch自带了nn.Conv2d，但开发者通常会写这样一个包装函数，目的是简化参数输入并固定特定的配置。
# 如果不写这个函数，你每次定义一层都要写：nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)。
# 有了这个函数，代码变得非常简洁：conv3x3(64, 128, stride=2)。
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)

# 这段代码是一个权重初始化函数。在深度学习中，模型刚创建时的初始参数（权重和偏置）不能是随机乱填的，否则网络可能由于梯度爆炸或梯度消失而无法训练。
# 这个函数的作用是：遍历模型的每一层，根据层的类型（卷积层或归一化层），应用特定的数学分布来初始化参数。
def conv_init(m):
# .__class__: 获取这个对象所属的类。
# .__name__: 获取这个类的名称（返回一个字符串）。
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

# 这段代码定义了 Wide ResNet (宽残差网络) 的核心构建块：wide_basic。
# Wide ResNet的核心思想是：与其单纯地增加网络的深度（层数），不如增加网络的宽度（通道数）。这种结构在处理域泛化（Domain Generalization）任务时非常流行，因为它在保持强大特征提取能力的同时，比极深的模型更容易训练。
class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

# 这是残差网络的核心。如果输入和输出的尺寸、通道数完全一样，shortcut就是一个恒等映射（Identity），直接把输入加到输出上。
# 如果尺寸变了（stride=2）或者通道数增加了，就需要用一个1x1卷积来调整输入的维度，确保它能和conv2的输出进行“加法”运算。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride,
                    bias=True), )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    """Wide Resnet with the softmax layer chopped off"""
    def __init__(self, input_shape, depth, widen_factor, dropout_rate):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
# WRN由1个初始卷积层、3个主要的Stage（每个Stage有n个wide_basic块，每个块含2层卷积）和1个最后的分类层组成构成(在该WRN实现中，分类层被切掉了，分类层被移到了predict（）函数中)。计算公式为：1 + (n * 2) * 3 + 1 = 6n+2。再加上某些结构差异，标准WRN定义为6n+4。通常最后两层是平均池化层。
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
# n代表了每一个Stage中包含的wide_basic模块的数量
        n = (depth - 4) / 6
# k(widen_factor)：这是控制“宽度”的系数。比如常用的WRN-28-10，意味着深度是28，k=10。相比于普通ResNet，它的通道数扩大了10倍.
        k = widen_factor

# 这行代码定义了Wide ResNet四个不同阶段（Stage）的输出通道数（Channels）。
# nStages[0]的输出通道数是16，分配给self.conv1，剩下的以此类推。
        # print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(input_shape[0], nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

        self.n_outputs = nStages[3]

        self.activation = nn.Identity() # for URM; does not affect other algorithms

# 这个函数的任务是根据你的深度要求，批量生产出一组wide_basic模块（也就是残差块），并将它们串联成一个完整的阶段（Stage）。
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
# 决定了在一个阶段内，哪个块负责“缩小图片尺寸”，哪些块负责“特征加工”。
# 第一个块 (stride)：只有Stage的第一个块会使用传入的stride（通常是 1 或 2）。如果stride=2，这个块就会把图像的长宽各缩减一半。
# 后续块([1] * ...)：同一个Stage剩下的所有块，步长全部固定为1。这意味着它们只负责加深特征提取，而不改变图像尺寸。
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
# *layers (解包操作)：这是一个Python技巧。layers是一个包含多个模块对象的列表，*号把列表里的每一个元素拆出来，作为独立的参数传给nn.Sequential。
# 在Python中，当你在调用函数时使用*，它会将一个列表（List）或元组（Tuple）中的所有元素平铺开来，变成一个个独立的参数传给函数。
# 正常写法是nn.Sequential(block1, block2, block3)，如果你直接写nn.Sequential(layers)，PyTorch会报错。因为它会认为你只传了一个参数（这个参数是个列表），而它想要的是一堆层。
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out[:, :, 0, 0]
        out = self.activation(out)
        return out