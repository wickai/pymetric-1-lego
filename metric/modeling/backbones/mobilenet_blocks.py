import torch.nn as nn


# from config import args
class Args:
    def __init__(self):
        self.use_se = False


args = Args()

blocks_keys = [
    "mobilenet_3x3_ratio_3",
    "mobilenet_3x3_ratio_6",
    "mobilenet_5x5_ratio_3",
    "mobilenet_5x5_ratio_6",
    "mobilenet_7x7_ratio_3",
    "mobilenet_7x7_ratio_6",
]

interverted_residual_setting = [
    [6, 32, 4, 2],
    [6, 56, 4, 2],
    [6, 112, 4, 2],
    [6, 128, 4, 1],
    [6, 256, 4, 2],
    [6, 432, 1, 1],
]

#################


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3), keepdim=True)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.hardsigmoid(y)
        return x * y


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, ksize, padding, stride, expand_ratio, use_se):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.use_se = args.use_se

        hidden_dim = inp * expand_ratio
        layers = []
        if expand_ratio != 1:
            # Expand
            layers.extend(
                [
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )
        # Depthwise Convolution
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    ksize,
                    stride,
                    padding,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        )
        # SE Block
        if self.use_se:
            layers.append(SEBlock(hidden_dim))
        # Pointwise Convolution
        layers.extend(
            [
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 定义操作字典
blocks_dict = {
    "mobilenet_3x3_ratio_3": lambda inp, oup, stride: InvertedResidual(
        inp, oup, 3, 1, stride, 3, use_se=args.use_se
    ),
    "mobilenet_3x3_ratio_6": lambda inp, oup, stride: InvertedResidual(
        inp, oup, 3, 1, stride, 6, use_se=args.use_se
    ),
    "mobilenet_5x5_ratio_3": lambda inp, oup, stride: InvertedResidual(
        inp, oup, 5, 2, stride, 3, use_se=args.use_se
    ),
    "mobilenet_5x5_ratio_6": lambda inp, oup, stride: InvertedResidual(
        inp, oup, 5, 2, stride, 6, use_se=args.use_se
    ),
    "mobilenet_7x7_ratio_3": lambda inp, oup, stride: InvertedResidual(
        inp, oup, 7, 3, stride, 3, use_se=args.use_se
    ),
    "mobilenet_7x7_ratio_6": lambda inp, oup, stride: InvertedResidual(
        inp, oup, 7, 3, stride, 6, use_se=args.use_se
    ),
}
