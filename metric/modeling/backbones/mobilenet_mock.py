import torch.nn as nn
import random
from .mobilenet_blocks import (
    blocks_dict,
    blocks_keys,
    interverted_residual_setting,
)
from metric.core.config import cfg

# from config import args, blocks_keys


# RANS = best_rngs = [1, 1, 4, 3, 2, 2, 1, 3, 1, 4, 3, 4, 0, 2, 2, 4, 1, 5, 3, 4, 3]


class MobileNetMock(nn.Module):
    def __init__(
        self,
        # rngs,
        # interverted_residual_setting,
        # num_classes=args.num_classes,
        width_mult=1.0,
    ):
        super(MobileNetMock, self).__init__()
        self.blocks_keys = blocks_keys
        self.blocks_dict = blocks_dict
        self.interverted_residual_setting = [
            [t, int(c * width_mult), n, s]
            for t, c, n, s in interverted_residual_setting
        ]
        # self.num_classes = num_classes
        self.width_mult = width_mult

        # 初始卷积层
        input_channel = int(16 * width_mult)
        frist_layer_stride = cfg.MODEL.FIRST_LAYER_STRIDE
        self.conv_bn = nn.Sequential(
            nn.Conv2d(
                3,
                input_channel,
                kernel_size=3,
                stride=frist_layer_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        # 动态特征层
        RANS = cfg.MODEL.RANS
        self.features, output_channel = self._make_features(input_channel, RANS)

        # 分类层
        last_channel = int(960 * width_mult)
        self.conv_1x1_bn = nn.Sequential(
            nn.Conv2d(
                output_channel, last_channel, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(inplace=True),
        )
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(last_channel, num_classes)

    def _make_features(self, input_channel, rngs):
        layers = []
        idx = 0
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                block_key = self.blocks_keys[rngs[idx]]
                layers.append(
                    self.blocks_dict[block_key](input_channel, output_channel, stride)
                )
                input_channel = output_channel
                idx += 1
        return nn.Sequential(*layers), input_channel

    def forward(self, x):
        x = self.conv_bn(x)
        x = self.features(x)
        x = self.conv_1x1_bn(x)
        # x = self.pool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x

    # 动态生成模型的方法
    def get_model(self, rngs):
        return MobileNetMock(
            rngs=rngs,
            interverted_residual_setting=self.interverted_residual_setting,
            # num_classes=self.num_classes,
            width_mult=self.width_mult,
        )

    @staticmethod
    def random_rngs(interverted_residual_setting):
        num_blocks = sum([n for _, _, n, _ in interverted_residual_setting])
        return [random.randint(0, len(blocks_keys) - 1) for _ in range(num_blocks)]

    @staticmethod
    def mutate_rngs(rngs, mutation_rate=0.3):
        mutated_rngs = rngs[:]
        for i in range(len(mutated_rngs)):
            if random.random() < mutation_rate:
                mutated_rngs[i] = random.randint(0, len(blocks_keys) - 1)
        return mutated_rngs
