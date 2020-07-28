import torch.nn as nn
import math
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../../')))
from devkit.ops import SwitchNorm2d


__all__ = ['ResNetV2SN', 'resnetv2sn18', 'resnetv2sn34', 'resnetv2sn50', 'resnetv2sn101',
           'resnetv2sn152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, using_moving_average=True, using_bn=True):
        super(BasicBlock, self).__init__()
        self.sn1 = SwitchNorm2d(inplanes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.sn2 = SwitchNorm2d(planes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.sn1(x)
        out_preact = self.relu(out)
        out = self.conv1(out_preact)

        out = self.sn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(out_preact)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, using_moving_average=False, using_bn=True):
        super(Bottleneck, self).__init__()
        self.sn1 = SwitchNorm2d(inplanes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.sn2 = SwitchNorm2d(planes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.sn3 = SwitchNorm2d(planes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.sn1(x)
        out_preact = self.relu(out)
        out = self.conv1(out_preact)

        out = self.sn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.sn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(out_preact)

        out += residual

        return out


class ResNetV2SN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, using_moving_average=True, using_bn=True):
        self.inplanes = 64
        self.using_moving_average=using_moving_average
        self.using_bn = using_bn
        super(ResNetV2SN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.sn1 = SwitchNorm2d(64, using_moving_average=self.using_moving_average, using_bn=self.using_bn)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.sn_out = SwitchNorm2d(512 * block.expansion, using_moving_average=self.using_moving_average, using_bn=self.using_bn)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.drouput = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SwitchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,using_moving_average=self.using_moving_average, using_bn=self.using_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, using_moving_average=self.using_moving_average, using_bn=self.using_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.sn_out(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.drouput(x)
        x = self.fc(x)

        return x


def resnetv2sn18(**kwargs):
    """Constructs a ResNetV2SN-18 model using switchable normalization.
    """
    model = ResNetV2SN(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnetv2sn34(**kwargs):
    """Constructs a ResNetV2SN-34 model using switchable normalization.
    """
    model = ResNetV2SN(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnetv2sn50(**kwargs):
    """Constructs a ResNetV2SN-50 model using switchable normalization.
    """
    model = ResNetV2SN(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnetv2sn101(**kwargs):
    """Constructs a ResNetV2SN-101 model using switchable normalization.
    """
    model = ResNetV2SN(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnetv2sn152(**kwargs):
    """Constructs a ResNetV2SN-152 model using switchable normalization.
    """
    model = ResNetV2SN(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
