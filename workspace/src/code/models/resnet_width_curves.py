import math
import torch.nn as nn

import curves
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride,
                         padding=1, bias=False)

class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, fix_points, stride=1):
        super(BasicBlockCurve, self).__init__()
        self.conv1 = curves.Conv2d(in_planes, planes, fix_points=fix_points, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes, fix_points=fix_points, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)

        shortcut = []
        if stride != 1 or in_planes != self.expansion*planes:
            shortcut = [
                curves.Conv2d(in_planes, self.expansion*planes, fix_points=fix_points, kernel_size=1, stride=stride, bias=False),
                curves.BatchNorm2d(self.expansion*planes, fix_points=fix_points)
            ]
        self.shortcut = nn.ModuleList(shortcut)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        for block in self.shortcut:
            x = block(x)
        residual = x
        out += residual
        out = F.relu(out)
        return out


class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, fix_points, stride=1):
        super(BottleneckCurve, self).__init__()
        self.conv1 = curves.Conv2d(in_planes, planes, fix_points=fix_points, kernel_size=1, bias=False)
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes, fix_points=fix_points, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv3 = curves.Conv2d(planes, self.expansion*planes, fix_points=fix_points, kernel_size=1, bias=False)
        self.bn3 = curves.BatchNorm2d(self.expansion*planes, fix_points=fix_points)

        shortcut = []
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = [
                curves.Conv2d(in_planes, self.expansion*planes, fix_points=fix_points, kernel_size=1, stride=stride, bias=False),
                curves.BatchNorm2d(self.expansion*planes, fix_points=fix_points)
            ]
        self.shortcut = nn.ModuleList(shortcut)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        for block in self.shortcut:
            x = block(x)
        residual = x
        out += residual
        out = F.relu(out)
        return out


class ResNetCurve(nn.Module):
    def __init__(self, block, num_blocks, fix_points, num_classes=10, k=1):
        super(ResNetCurve, self).__init__()
        self.in_planes = 1*k

        self.conv1 = curves.Conv2d(3, 1*k, kernel_size=3, stride=1, padding=1, bias=False, fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(1*k, fix_points=fix_points)
        self.layer1 = self._make_layer(block, 1*k, num_blocks[0], stride=1, fix_points=fix_points)
        self.layer2 = self._make_layer(block, 2*k, num_blocks[1], stride=2, fix_points=fix_points)
        self.layer3 = self._make_layer(block, 4*k, num_blocks[2], stride=2, fix_points=fix_points)
        self.layer4 = self._make_layer(block, 8*k, num_blocks[3], stride=2, fix_points=fix_points)
        self.linear = curves.Linear(8*k*block.expansion, num_classes, fix_points=fix_points)

    def _make_layer(self, block, planes, num_blocks, stride, fix_points=[]):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, fix_points, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)
        for block in self.layer4:
            out = block(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18Curve(num_classes, fix_points, width=1):
    return ResNetCurve(BasicBlockCurve, [2,2,2,2], fix_points=fix_points, num_classes=num_classes, k=width)

def ResNet34Curve(fix_points):
    return ResNetCurve(BasicBlockCurve, [3,4,6,3], fix_points=fix_points)

def ResNet50Curve(fix_points):
    return ResNetCurve(BottleneckCurve, [3,4,6,3], fix_points=fix_points)

def ResNet101Curve(fix_points):
    return ResNetCurve(BottleneckCurve, [3,4,23,3], fix_points=fix_points)

def ResNet152Curve(fix_points):
    return ResNetCurve(BottleneckCurve, [3,8,36,3], fix_points=fix_points)

# test()



