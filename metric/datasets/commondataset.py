#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import metric.core.logging as logging
import metric.datasets.transforms as transforms
import torch.utils.data
from metric.core.config import cfg
from PIL import Image
import random


logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
# imagenet
# _MEAN = [0.406, 0.456, 0.485]
# _SD = [0.225, 0.224, 0.229]

# food101
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
_MEAN = [0.485, 0.456, 0.406]
_SD = [0.229, 0.224, 0.225]

# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)



    
class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        logger.info("Constructing dataset from {}...".format(split))
        self._data_path, self._split = data_path, split
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        self._imdb, self._class_ids = [], []
        with open(os.path.join(self._data_path, self._split), "r") as fin:
            for line in fin:
                im_dir, cont_id = line.strip().split(" ")
                im_path = os.path.join(self._data_path, im_dir)
                self._imdb.append({"im_path": im_path, "class": int(cont_id)})
                self._class_ids.append(int(cont_id))
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(set(self._class_ids))))

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        # Train and test setups differ
        train_size = cfg.TRAIN.IM_SIZE
        if "train" in self._split:
            # Scale and aspect ratio then horizontal flip
            im = transforms.random_sized_crop(im=im, size=train_size, area_frac=0.08)
            im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        else:
            # Scale and center crop
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(train_size, im)
        # HWC -> CHW
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # PCA jitter
        if "train" in self._split:
            im = transforms.lighting(im, 0.1, _EIG_VALS, _EIG_VECS)
        # Color normalization
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        # Load the image
        try:
            im = cv2.imread(self._imdb[index]["im_path"])
            im = im.astype(np.float32, copy=False)
        except:
            print('error: ', self._imdb[index]["im_path"])
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)

class HuggingFaceImageNetDataset(DataSet):
    def __init__(self, hf_dataset, split):
        self._hf_dataset = hf_dataset
        self._split = split
        self._construct_imdb()

    def _construct_imdb(self):
        """Adapt HuggingFace dataset to your format."""
        self._imdb, self._class_ids = [], []
        # for example in self._hf_dataset:
        #     self._imdb.append({"image": example["image"], "class": example["label"]})
        #     self._class_ids.append(example["label"])
        self._imdb = list(range(len(self._hf_dataset)))  # 保存索引
        self._class_ids = list(range(1000))#[self._hf_dataset[i]["label"] for i in self._imdb]

        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(set(self._class_ids))))

    def __getitem__(self, index):
        ###### old
        # # Load image (already PIL Image)
        # # im = self._imdb[index]["image"]
        # # im = np.array(im).astype(np.float32)
        # data = self._hf_dataset[self._imdb[index]]
        # im = np.array(data["image"]).astype(np.float32)
        # # Apply transformation
        # # print("im.shape",im.shape)
        # if im.ndim == 2:
        #     im = np.stack([im] * 3, axis=-1)  # (H, W) -> (H, W, 3)
        # elif im.shape[2] == 1:
        #     im = np.repeat(im, 3, axis=2)     # (H, W, 1) -> (H, W, 3)
        # im = self._prepare_im(im)
        # # label = self._imdb[index]["class"]
        # label = data["label"]
        # return im, label
        
        # data = self._hf_dataset[self._imdb[index]]
        # print("DEBUG data keys:", data.keys())
        # print("DEBUG data:", data)
        ## add try
        max_retry = 10
        retry = 0
        while retry < max_retry:
            try:
                data = self._hf_dataset[self._imdb[index]]
                im = np.array(data["image"]).astype(np.float32)

                # 修正通道数
                if im.ndim == 2:
                    im = np.stack([im] * 3, axis=-1)  # (H, W) -> (H, W, 3)
                elif im.shape[2] == 1:
                    im = np.repeat(im, 3, axis=2)     # (H, W, 1) -> (H, W, 3)

                im = self._prepare_im(im)
                label = data["label"]
                return im, label

            except Exception as e:
                print(f"[Warning] Failed to load sample at index {index}: {e}")
                # import random
                index = random.randint(0, len(self._imdb) - 1)
                retry += 1

        raise RuntimeError(f"[Error] Repeatedly failed to load a valid sample after {max_retry} retries.")


# class HuggingFaceFood101Dataset(torch.utils.data.Dataset):
class HuggingFaceFood101Dataset(DataSet):
    def __init__(self, hf_dataset, split="train", transform=None):
        # 加载 food101 数据集
        self.dataset = hf_dataset
        self.transform = transform
        self._imdb = list(range(len(self.dataset)))
        self._class_ids = list(range(101))
        self._split = split
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(set(self._class_ids))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取图片路径和标签
        
        
        max_retry = 10
        retry = 0
        while retry < max_retry:
            try:
                # print(type(self.dataset[idx]["image"]))
                # print(self.dataset[idx]["image"], "label", self.dataset[idx]["label"])
                image = self.dataset[idx]["image"]
                label = self.dataset[idx]["label"]

                # 打开图片
                image = image.convert("RGB")
                image = np.array(image)
                image = self._prepare_im(image)
                # # 应用变换
                # if self.transform:
                #     image = self.transform(image)
                

                return image, label
        
            except Exception as e:
                print(f"[Warning] Failed to load sample at index {idx}: {e}")
                # import random
                idx = random.randint(0, len(self._imdb) - 1)
                retry += 1

        raise RuntimeError(f"[Error] Repeatedly failed to load a valid sample after {max_retry} retries.")