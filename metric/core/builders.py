#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

import torch
from metric.core.config import cfg

# from metric.modeling.backbones.resnet import ResNet

# from metric.modeling.backbones import VisionTransformer
# from metric.modeling.backbones import resnest269, resnest50
from metric.modeling.backbones.resnet_wider import resnet50x1
from metric.modeling.backbones.mobilenet_mock import MobileNetMock
from metric.modeling.backbones.mobilenet_supermodel import MobileNetSupermodel
from metric.modeling.backbones.mobilenetv2_nas import mobilenet_v2_nas
from metric.modeling.heads import LinearHead

# Supported backbones
_models = {
    # "resnet": ResNet,
    #    "vit": VisionTransformer,
    # "resnest269": resnest269,
    # "resnest50": resnest50,
    "resnet50x1": resnet50x1,
    # "resnet152x1": resnet152x1,
    "mobilenetmock": MobileNetMock,
    "mobilenet_supermodel": MobileNetSupermodel,
    "mobilenetv2_nas": mobilenet_v2_nas
}
# Supported loss functions
_loss_funs = {"cross_entropy": torch.nn.CrossEntropyLoss}
# Supported heads
_heads = {"LinearHead": LinearHead}


class MetricModel(torch.nn.Module):
    def __init__(self):
        super(MetricModel, self).__init__()
        self.backbone = build_model()
        self.head = build_head()

    def forward(self, x, targets):
        features = self.backbone(x)
        return self.head(features, targets=targets)


class SuperModel(torch.nn.Module):
    def __init__(self):
        super(SuperModel, self).__init__()
        self.backbone = build_model()
        self.head = build_head()

    def forward(self, x, targets, rngs):
        features = self.backbone(x, rngs)
        return self.head(features, targets=targets)


def get_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_head():
    err_str = "Head type '{}' not supported"
    assert cfg.MODEL.HEADS.NAME in _heads.keys(), err_str.format(cfg.MODEL.HEADS.NAME)
    return _heads[cfg.MODEL.HEADS.NAME]


def get_loss_fun():
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSSES.NAME in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSSES.NAME]


def build_arch():
    if cfg.MODEL.TYPE.endswith("_supermodel"):
        architecture = SuperModel()
    else:
        architecture = MetricModel()
    return architecture


def build_model():
    """Builds the model."""
    return get_model()()


def build_loss_fun():
    """Build the loss function."""
    sm = cfg.MODEL.LOSSES.LABEL_SMOOTHING
    return get_loss_fun()(label_smoothing=sm)


def build_head():
    """Build the head"""
    return get_head()()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_head(name, ctor):
    """Registers a head dynamically."""
    _heads[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
