import torch.nn as nn
class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class ResNet9(nn.Module):
    def __init__(self, NUM_CLASSES=10):
        super(ResNet9, self).__init__()
        self.model = nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, NUM_CLASSES, bias=False),
            Mul(0.2),
        )

    def forward(self, x):
        return self.model(x)
