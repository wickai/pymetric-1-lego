import torch
import torch.nn as nn
import random
import json
import os
import logging
from metric.core.config import cfg

# ============ MobileNet Components ============

class SEBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        rc = max(1, c // r)
        self.fc = nn.Sequential(
            nn.Linear(c, rc, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rc, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, *_ = x.shape
        y = self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class Zero(nn.Module):
    def __init__(self, stride, out_c):
        super().__init__()
        self.stride, self.out_c = stride, out_c

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if x.size(1) != self.out_c:
            pad = self.out_c - x.size(1)
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
        return x.mul(0.0)


class MBConv(nn.Module):
    def __init__(self, inp, oup, k, s, expand, se=False):
        super().__init__()
        self.use_res = (s == 1 and inp == oup)
        hid = inp * expand

        layers = []
        if expand != 1:
            layers += [
                nn.Conv2d(inp, hid, 1, bias=False),
                nn.BatchNorm2d(hid),
                nn.ReLU6(inplace=True)
            ]
        layers += [
            nn.Conv2d(hid, hid, k, s, k // 2, groups=hid, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU6(inplace=True)
        ]
        if se:
            layers.append(SEBlock(hid))
        layers += [
            nn.Conv2d(hid, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


# ============ Helper functions for pickle compatibility ============
def mb_builder(k, r, se, i, o, s, t):
    return MBConv(i, o, k, s, r, se)

def skip_builder(i, o, s, t):
    if s == 1 and i == o:
        return nn.Identity()
    else:
        return nn.Sequential(
            nn.Conv2d(i, o, 1, stride=s, bias=False),
            nn.BatchNorm2d(o)
        )

def zero_builder(i, o, s, t):
    return Zero(s, o)


class MobileNetV2(nn.Module):
    """
    Supports limit_blocks (prefix truncation):
    - limit_blocks=None: Build full network
    - limit_blocks=k: Build features for first k blocks, then head+pool+fc
    """
    def __init__(self, op_codes, width_codes, stage_setting, op_list, width_choices,
                 num_classes=10, small_input=True, limit_blocks=None):
        super().__init__()
        self._build_ops(op_list)

        stem_c = 16
        stem_stride = 1 if small_input else 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_c, 3, stem_stride, 1, bias=False),
            nn.BatchNorm2d(stem_c),
            nn.ReLU6(inplace=True),
        )

        # Build features block by block, respecting limit_blocks
        layers = []
        in_c, blk_idx = stem_c, 0
        total_blocks = sum(s[2] for s in stage_setting)
        max_blocks = total_blocks if limit_blocks is None else max(0, min(limit_blocks, total_blocks))

        for stage_idx, (t, c_base, n, s) in enumerate(stage_setting):
            if blk_idx >= max_blocks:
                break
            out_c = int(round(c_base * width_choices[width_codes[stage_idx]]))
            for i in range(n):
                if blk_idx >= max_blocks:
                    break
                stride = s if i == 0 else 1
                op_name = op_list[op_codes[blk_idx]]
                layers.append(self._op_factory(op_name, in_c, out_c, stride, t))
                in_c, blk_idx = out_c, blk_idx + 1
        self.features = nn.Sequential(*layers)

        last_c = 1024 if small_input else 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_c, last_c, 1, bias=False),
            nn.BatchNorm2d(last_c),
            nn.ReLU6(inplace=True),
        )
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(last_c, num_classes)

    def _build_ops(self, op_list):
        # Use partial instead of lambda to support pickle
        from functools import partial

        self._ops = {}
        for k in (3, 5):
            for r in (1, 2, 4, 6):
                base = f"mbconv_{k}x{k}_r{r}"
                self._ops[base] = partial(mb_builder, k, r, False)
                self._ops[base + "_se"] = partial(mb_builder, k, r, True)

        self._ops["skip_connect"] = skip_builder
        self._ops["zero"] = zero_builder
        self.op_list = list(op_list)

    def _op_factory(self, name, i, o, s, t):
        return self._ops[name](i, o, s, t)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        # x = self.pool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x


class MobileNetSearchSpace:
    _STAGE_SETTING = [
        # t, c, n, s
        [1,   16, 1, 1],
        [6,   24, 2, 1],
        [6,   32, 3, 1],
        [6,   64, 4, 2],  # ↓8×8
        [6,   96, 4, 1],
        [6,  160, 5, 2],  # ↓4×4
        [6,  320, 2, 1],
    ]
    _WIDTH_CHOICES = [0.5, 0.75, 1.0]

    @staticmethod
    def _default_op_list():
        ops = []
        for k in (3, 5):
            for r in (1, 2, 4, 6):
                ops.append(f"mbconv_{k}x{k}_r{r}")
                ops.append(f"mbconv_{k}x{k}_r{r}_se")
        ops.append("skip_connect")
        return ops

    def __init__(self, num_classes=10, small_input=True):
        self.stage_setting = self._STAGE_SETTING
        self.op_list = self._default_op_list()
        self.width_choices = self._WIDTH_CHOICES
        self.total_blocks = sum(s[2] for s in self.stage_setting)
        self.num_classes = num_classes
        self.small_input = small_input

    def get_model(self, op_codes, width_codes):
        # Full model
        return MobileNetV2(
            op_codes=op_codes,
            width_codes=width_codes,
            stage_setting=self.stage_setting,
            op_list=self.op_list,
            width_choices=self.width_choices,
            num_classes=self.num_classes,
            small_input=self.small_input,
            limit_blocks=None
        )

    def get_prefix_model(self, op_prefix, width_codes):
        # Prefix model
        return MobileNetV2(
            op_codes=op_prefix,
            width_codes=width_codes,
            stage_setting=self.stage_setting,
            op_list=self.op_list,
            width_choices=self.width_choices,
            num_classes=self.num_classes,
            small_input=self.small_input,
            limit_blocks=len(op_prefix)
        )


def mobilenetv2_legonas_lw():
    """
    Constructs a MobileNetV2 model based on a NAS-searched JSON architecture.
    """
    if hasattr(cfg.MODEL, "ARCH_JSON") and cfg.MODEL.ARCH_JSON:
        arch_json_path = cfg.MODEL.ARCH_JSON
    else:
        # Fallback or error? Assuming user provides it in config.
        # If not in cfg, maybe look for it in a default place or raise error.
        raise ValueError("cfg.MODEL.ARCH_JSON must be specified.")

    if not os.path.exists(arch_json_path):
        raise FileNotFoundError(f"Architecture JSON not found at {arch_json_path}")
    
    logging.info(f"Loading architecture from {arch_json_path}")
    with open(arch_json_path, 'r') as f:
        arch_data = json.load(f)
    
    op_codes = arch_data['op_codes']
    width_codes = arch_data['width_codes']
    
    # Heuristic for small input: check if IM_SIZE is small (e.g. CIFAR 32x32)
    # ImageNet is 224, so it is NOT small_input.
    is_small_input = cfg.TRAIN.IM_SIZE < 100
    
    sp = MobileNetSearchSpace(
        num_classes=cfg.MODEL.NUM_CLASSES, 
        small_input=is_small_input
    )
    
    if len(op_codes) < sp.total_blocks:
        logging.info(f"Building PREFIX model with {len(op_codes)} blocks.")
        model = sp.get_prefix_model(op_codes, width_codes)
    else:
        logging.info("Building FULL model.")
        model = sp.get_model(op_codes, width_codes)
        
    return model
