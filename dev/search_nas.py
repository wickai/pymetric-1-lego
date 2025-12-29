#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import time
import torch
from torchvision import datasets, transforms
import sys
from datasets import load_from_disk, load_dataset
from PIL import Image

# Import from common
# Assuming nas_common.py is in the same directory
from nas_common import (
    set_seed, setup_logger, MobileNetSearchSpace, SWAP,
    LayerwiseBlockES, EvolutionarySearch, count_parameters_in_MB
)

def load_imagenet_dataset(split: str):
    if split == 'validation':
        split = 'val'
    # 关键路径：尝试从本地固定路径加载 Arrow 格式的 ImageNet 数据
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

class HuggingFaceImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, split='train', transform=None):
        self.hf_dataset = hf_dataset
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def parse_args():
    p = argparse.ArgumentParser("MobileNetV2 Search with SWAP (Search Only)")
    p.add_argument("--log_path", default="./logs", type=str)
    p.add_argument("--log_name", default="search_nas.log", type=str)
    p.add_argument("--data_path", default="./data", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--output_path", default="./best_arch.json", type=str, help="Path to save the best architecture JSON")

    # Dataset
    p.add_argument("--dataset", choices=["cifar10", "imagenet"], default="cifar10", help="Dataset to use for search")

    # Search common
    p.add_argument("--search_mode", choices=["layer_ea", "global_ea"], default="layer_ea")
    p.add_argument("--search_batch", default=64, type=int)
    p.add_argument("--num_inits", default=1, type=int)
    p.add_argument("--small_input", action="store_true", default=True)
    p.add_argument("--num_classes", default=10, type=int)

    # Global EA
    p.add_argument("--population_size", default=20, type=int)
    p.add_argument("--mutation_rate", default=0.3, type=float)
    p.add_argument("--n_generations", default=50, type=int)

    # Layer-wise EA
    p.add_argument("--n_blocks_to_search", default=None, type=int)
    p.add_argument("--layer_population", default=16, type=int)
    p.add_argument("--layer_generations", default=8, type=int)
    p.add_argument("--layer_mutation", default=0.3, type=float)
    p.add_argument("--stagewise_width_search", action="store_true", default=True)
    p.add_argument("--use_pareto", action="store_true", default=True, help="使用 pareto front 多目标优化 (SWAP ↑, ParamsMB ↓)")
    
    return p.parse_args()

def main():
    args = parse_args()
    setup_logger(args.log_path, args.log_name)
    logging.info("Args:\n" + json.dumps(vars(args), indent=4))
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Determine num_classes and small_input based on dataset
    if args.dataset == 'imagenet':
        num_classes = 1000
        small_input = False
    else:
        num_classes = args.num_classes
        small_input = args.small_input

    # Search Space & SWAP
    sp = MobileNetSearchSpace(num_classes=num_classes, small_input=small_input)
    swap = SWAP(device=device)

    # Prepare search input
    if args.dataset == 'cifar10':
        search_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010]),
        ])
        # Note: Use train=True for search as in original code
        search_ds = datasets.CIFAR10(args.data_path, train=True, download=True, transform=search_transform)
    elif args.dataset == 'imagenet':
        search_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Use train split for search
        hf_dataset = load_imagenet_dataset('train')
        search_ds = HuggingFaceImageNetDataset(hf_dataset, split='train', transform=search_transform)

    search_loader = torch.utils.data.DataLoader(search_ds, batch_size=args.search_batch,
                                                shuffle=False, num_workers=2, pin_memory=True)
    mini_inputs, _ = next(iter(search_loader))
    mini_inputs = mini_inputs.to(device)

    # Search
    t0 = time.time()
    if args.search_mode == "layer_ea":
        les = LayerwiseBlockES(
            search_space=sp, swap_metric=swap, device=device, num_inits=args.num_inits,
            population_size=args.layer_population, mutation_rate=args.layer_mutation,
            n_generations=args.layer_generations, stagewise_width_search=args.stagewise_width_search,
            use_pareto=args.use_pareto
        )
        best = les.search(mini_inputs, n_blocks_to_search=args.n_blocks_to_search)
    else:
        es = EvolutionarySearch(
            population_size=args.population_size, mutation_rate=args.mutation_rate,
            n_generations=args.n_generations, swap_metric=swap, search_space=sp,
            device=device, num_inits=args.num_inits
        )
        best = es.search(mini_inputs)
    t1 = time.time()
    
    # Ensure params_mb is present
    if "params_mb_prefix" not in best:
        # Fallback for Global EA or if params_mb_prefix is missing
        if "op_codes" in best:
             model = sp.get_model(best["op_codes"], best["width_codes"])
             best["params_mb_prefix"] = count_parameters_in_MB(model) # Note: this might be full model params
             best["op_codes_prefix"] = best["op_codes"]
             best["fitness"] = best["fitness"]
    
    logging.info(f"Search finished in {t1 - t0:.2f}s.")
    logging.info(f"Best architecture | SWAP fitness={best['fitness']:.3f}")
    logging.info(f"Best architecture | op_codes_prefix={best['op_codes_prefix']}")
    logging.info(f"Best architecture | width_codes={best['width_codes']}")
    logging.info(f"Best Model param: {best.get('params_mb_prefix', 0):.2f} MB")

    # Save to JSON
    best_arch = {
        "op_codes": best.get("op_codes_prefix"),
        "width_codes": best["width_codes"],
        "fitness": best["fitness"],
        "params_mb": best.get("params_mb_prefix")
    }
    
    with open(args.output_path, "w") as f:
        json.dump(best_arch, f, indent=4)
    logging.info(f"Best architecture saved to {args.output_path}")

if __name__ == "__main__":
    main()
