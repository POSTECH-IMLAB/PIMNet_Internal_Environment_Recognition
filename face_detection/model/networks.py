import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv3_bn_lrelu(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv1_bn(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    # SSH: Single Stage Headless Face Detector
    # https://arxiv.org/abs/1708.03979
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.conv3X3 = conv3_bn(in_channels, out_channels//2, stride=1)

        self.conv5X5_1 = conv3_bn_lrelu(in_channels, out_channels//4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv3_bn(out_channels//4, out_channels//4, stride=1)

        self.conv7X7_2 = conv3_bn_lrelu(out_channels//4, out_channels//4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv3_bn(out_channels//4, out_channels//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, width=0.25):
        super().__init__()
        self.stage1 = nn.Sequential(
            conv3_bn_lrelu(3, round(width*32), 2, leaky=0.1), # 3
            conv_dw(round(width*32), round(width*64), 1),    # 7
            conv_dw(round(width*64), round(width*128), 2),   # 11
            conv_dw(round(width*128), round(width*128), 1),  # 19
            conv_dw(round(width*128), round(width*256), 2),  # 27
            conv_dw(round(width*256), round(width*256), 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(round(width*256), round(width*512), 2),  # 43 + 16 = 59
            conv_dw(round(width*512), round(width*512), 1),  # 59 + 32 = 91
            conv_dw(round(width*512), round(width*512), 1),  # 91 + 32 = 123
            conv_dw(round(width*512), round(width*512), 1),  # 123 + 32 = 155
            conv_dw(round(width*512), round(width*512), 1),  # 155 + 32 = 187
            conv_dw(round(width*512), round(width*512), 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(round(width*512), round(width*1024), 2),  # 219 +3 2 = 241
            conv_dw(round(width*1024), round(width*1024), 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x).view(-1, 256)
        x = self.fc(x)
        return x
