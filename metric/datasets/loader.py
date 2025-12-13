#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os
os.environ["HF_DATASETS_CACHE"] = "/data/wk/kai/data/hf_cache"

import torch
from metric.core.config import cfg
from metric.datasets.commondataset import DataSet, HuggingFaceImageNetDataset, HuggingFaceFood101Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from .cifar10 import Cifar10
from metric.datasets.torch_transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
import torchvision
from datasets import load_dataset, load_from_disk


# Default data directory (/path/pycls/pycls/datasets/data)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Relative data paths to default data directory
_PATHS = {"cifar10": "cifar10"}


def load_imagenet_dataset(split: str):
    if split == 'validation':
        split = 'val'
    local_path = os.path.join("/data/wk/kai/data/imagenet_arrow", split)
    print("local_path", local_path)
    if os.path.exists(local_path):
        print(f"[✓] 使用本地数据集: {local_path}")
        hf_dataset = load_from_disk(local_path)
    else:
        print("[⭳] 本地数据集不存在，正在从 Hugging Face Hub 加载...")
        # hf_dataset = load_dataset("imagenet-1k", split=split)
        pass
    
    return hf_dataset

def load_food101_dataset(split: str):
    hf_dataset = load_dataset(
        "parquet",
        data_files={
            "train": "/data/wk/kai/data/datasets--ethz--food101/data/train-*.parquet",
            "validation": "/data/wk/kai/data/datasets--ethz--food101/data/validation-*.parquet"
        },
        split=split
    )
    return hf_dataset

def _construct_loader(
    dataset_name, split, batch_size, shuffle, drop_last, use_mixup_cutmix
):
    """Constructs the data loader for the given dataset."""
    if dataset_name.lower() == "cifar10":
        data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name.lower()])
        dataset = Cifar10(data_path, split)
    elif dataset_name.lower() == "imagenet":
        # hf_dataset = load_dataset("imagenet-1k", split=split)  # or "validation"
        # hf_dataset = load_dataset(
        #         path="/data/wk/kai/data/mirror/HuggingFace-Download-Accelerator/hf_hub/datasets--imagenet-1k", 
        #         data_dir="data",split=split)  # or "validation"
        hf_dataset = load_imagenet_dataset(split)
        dataset = HuggingFaceImageNetDataset(hf_dataset, split=split)
    elif dataset_name.lower() == "food101":
        hf_dataset = load_food101_dataset(split)
        if split == 'train':
            train_size = cfg.TRAIN.IM_SIZE
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((train_size, train_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            test_size = cfg.TEST.IM_SIZE
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((test_size, test_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        dataset = HuggingFaceFood101Dataset(hf_dataset=hf_dataset, split=split, transform=transform)
    else:
        data_path = os.path.join(_DATA_DIR, dataset_name)
        # Construct the dataset from commendataset
        dataset = DataSet(data_path, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    # mixup_cutmix
    num_classes = len(set(dataset._class_ids))
    if use_mixup_cutmix:
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=cfg.DATA_LOADER.MIXUP_ALPHA,
            cutmix_alpha=cfg.DATA_LOADER.CUTMIX_ALPHA,
            num_classes=num_classes,
            use_v2=False,
        )
    else:
        mixup_cutmix = None
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True,
        use_mixup_cutmix=True,
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
        use_mixup_cutmix=False,
    )


def shuffle(loader, cur_epoch):
    """ "Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
