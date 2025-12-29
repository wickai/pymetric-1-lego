#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import logging
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# ============ 1) 通用工具函数 ============

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred_topk = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_top1 += (pred_topk[:, 0] == labels).sum().item()
        for i in range(labels.size(0)):
            if labels[i].item() in pred_topk[i].tolist():
                correct_top5 += 1
        total += labels.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    return top1_acc, top5_acc


# ============ 2) Cutout（仅最终训练用） ============

class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            x1 = int(np.clip(x - self.length // 2, 0, w))
            x2 = int(np.clip(x + self.length // 2, 0, w))
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


# ============ 3) Mixup（仅最终训练用） ============

def mixup_data(x, y, alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    b = x.size(0)
    index = torch.randperm(b).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ 4) SWAP 评估 ============

class SampleWiseActivationPatterns:
    def __init__(self, device):
        self.device = device
        self.activations = None

    @torch.no_grad()
    def collect_activations(self, feats):
        self.activations = feats.sign().to(self.device)

    @torch.no_grad()
    def calc_swap(self):
        if self.activations is None:
            return 0
        self.activations = self.activations.t()  # (F, N)
        uniq = torch.unique(self.activations, dim=0).size(0)
        return int(uniq)


class SWAP:
    def __init__(self, device):
        self.device = device
        self.inter_feats = []
        self.swap_evaluator = SampleWiseActivationPatterns(device)

    def evaluate(self, model, inputs):
        hooks = []
        for _, m in model.named_modules():
            if isinstance(m, (nn.ReLU, nn.ReLU6)):
                hooks.append(m.register_forward_hook(self._hook_fn))
        self.inter_feats = []

        model.eval()
        with torch.no_grad():
            model(inputs.to(self.device))

        if len(self.inter_feats) == 0:
            for h in hooks: h.remove()
            return 0

        feats = torch.cat(self.inter_feats, dim=1)
        self.swap_evaluator.collect_activations(feats)
        score = self.swap_evaluator.calc_swap()

        for h in hooks: h.remove()
        self.inter_feats = []
        return float(score)

    def _hook_fn(self, module, inp, out):
        feats = out.detach().reshape(out.size(0), -1)
        self.inter_feats.append(feats)


# ============ 5) MobileNet（MBConv/SE/Skip投影） ============

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
    支持 limit_blocks（前缀截断构建）：
    - 当 limit_blocks=None：构建完整网络（原行为）
    - 当 limit_blocks=k：仅构建前 k 个 blocks 的 features，再接 head+pool+fc
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

        # 逐 block 构建 features，尊重 limit_blocks
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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_c, num_classes)

    def _build_ops(self, op_list):
        # 使用 partial 替代 lambda，以便支持 pickle
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
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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

    def random_op_codes(self):
        return [random.randrange(len(self.op_list)) for _ in range(self.total_blocks)]

    def random_width_codes(self):
        return [random.randrange(len(self.width_choices)) for _ in range(len(self.stage_setting))]

    def mutate_op_codes(self, codes, mutation_rate=0.3):
        mutated = codes[:]
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.randrange(len(self.op_list))
        return mutated

    def mutate_width_codes(self, codes, mutation_rate=0.3):
        mutated = codes[:]
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.randrange(len(self.width_choices))
        return mutated

    def get_model(self, op_codes, width_codes):
        # 完整模型（不截断）
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
        # 仅构建前缀（前 len(op_prefix) 个 blocks）
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


# ============ 6) 退化检测工具 ============

@torch.no_grad()
def is_degenerate_head(model, inputs, device, eps: float = 1e-6):
    model.eval()
    x = inputs.to(device)
    x = model.stem(x)
    x = model.features(x)
    x_head = model.head(x)  # [B, C, H, W]
    m = x_head.abs().mean().item()
    s = x_head.std().item()
    return (m < eps) or (s < eps)


# ============ 7) 全局 Evolutionary Search（保留以对比） ============

class EvolutionarySearch:
    def __init__(self, population_size, mutation_rate, n_generations,
                 swap_metric, search_space, device,
                 num_inits=1, eval_loader=None):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.swap_metric = swap_metric
        self.search_space = search_space
        self.device = device
        self.num_inits = num_inits
        self.eval_loader = eval_loader

    def _score(self, op_codes, width_codes, inputs):
        scores = []
        for _ in range(self.num_inits):
            model = self.search_space.get_model(op_codes, width_codes).to(self.device)
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p)
            if is_degenerate_head(model, inputs, self.device):
                scores.append(-1e12)
            else:
                s = self.swap_metric.evaluate(model, inputs)
                scores.append(s)
        return float(np.mean(scores))

    def search(self, inputs):
        pop = []
        for _ in range(self.population_size):
            pop.append({
                "op_codes": self.search_space.random_op_codes(),
                "width_codes": self.search_space.random_width_codes(),
                "fitness": None
            })

        for g in range(self.n_generations):
            logging.info(f"=== Generation {g+1}/{self.n_generations} ===")
            for i, ind in enumerate(pop):
                if ind["fitness"] is None:
                    fit = self._score(ind["op_codes"], ind["width_codes"], inputs)
                    ind["fitness"] = fit
                    logging.info(f"  [Ind-{i+1}] fitness(SWAP): {fit:.3f}")

            pop.sort(key=lambda x: x["fitness"], reverse=True)  # 仅按SWAP
            logging.info(f"  Best in Gen{g+1}: fitness={pop[0]['fitness']:.3f}")

            next_gen = pop[: self.population_size // 2]
            while len(next_gen) < self.population_size:
                p1, p2 = random.choice(next_gen), random.choice(next_gen)
                child_ops = self.crossover(p1["op_codes"], p2["op_codes"])
                child_wds = self.crossover(p1["width_codes"], p2["width_codes"])
                child_ops = self.search_space.mutate_op_codes(child_ops, self.mutation_rate)
                child_wds = self.search_space.mutate_width_codes(child_wds, self.mutation_rate)
                next_gen.append({"op_codes": child_ops, "width_codes": child_wds, "fitness": None})
            pop = next_gen

        for ind in pop:
            if ind["fitness"] is None:
                ind["fitness"] = self._score(ind["op_codes"], ind["width_codes"], inputs)

        pop.sort(key=lambda x: x["fitness"], reverse=True)
        return pop[0]

    @staticmethod
    def crossover(c1, c2):
        L = len(c1)
        p1 = random.randint(0, L - 1)
        p2 = random.randint(p1, L - 1)
        return c1[:p1] + c2[p1:p2] + c1[p2:]


# ============ 7.5) Pareto Front 工具函数 ============

def dominates_dict(a: dict, b: dict) -> bool:
    """
    a 是否支配 b
    目标1：SWAP ↑ (越大越好)
    目标2：ParamsMB ↓ (越小越好)
    """
    better_or_equal_swap = a["absolute_swap"] >= b["absolute_swap"]
    better_or_equal_params = a["params_mb"] <= b["params_mb"]
    strictly_better_one = (a["absolute_swap"] > b["absolute_swap"]) or (a["params_mb"] < b["params_mb"])
    return better_or_equal_swap and better_or_equal_params and strictly_better_one


def non_dominated_sort_dict(population: list) -> list:
    """
    对字典列表进行非支配排序，返回 fronts 列表
    每个个体字典需要包含 "absolute_swap" 和 "params_mb" 字段
    """
    S = {}
    n = {}
    fronts = [[]]

    for i, p in enumerate(population):
        S[i] = []
        n[i] = 0
        for j, q in enumerate(population):
            if i == j:
                continue
            if dominates_dict(p, q):
                S[i].append(j)
            elif dominates_dict(q, p):
                n[i] += 1
        if n[i] == 0:
            p["rank"] = 0
            fronts[0].append(p)

    idx = 0
    while fronts[idx]:
        next_front = []
        for p in fronts[idx]:
            p_idx = population.index(p)
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    population[q_idx]["rank"] = idx + 1
                    next_front.append(population[q_idx])
        idx += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance_dict(front: list) -> None:
    """
    计算拥挤距离（原地修改）
    每个个体字典需要包含 "absolute_swap" 和 "params_mb" 字段
    """
    if len(front) == 0:
        return
    if len(front) == 1:
        front[0]["crowding_distance"] = float("inf")
        return
    if len(front) == 2:
        front[0]["crowding_distance"] = float("inf")
        front[1]["crowding_distance"] = float("inf")
        return

    # 初始化
    for ind in front:
        ind["crowding_distance"] = 0.0

    # 目标1：swap_score (越大越好)
    front.sort(key=lambda x: x["absolute_swap"])
    front[0]["crowding_distance"] = float("inf")
    front[-1]["crowding_distance"] = float("inf")
    s_min, s_max = front[0]["absolute_swap"], front[-1]["absolute_swap"]
    s_range = s_max - s_min if s_max > s_min else 1e-9
    for i in range(1, len(front) - 1):
        front[i]["crowding_distance"] += (
            (front[i + 1]["absolute_swap"] - front[i - 1]["absolute_swap"]) / s_range
        )

    # 目标2：params_mb（越小越好）
    front.sort(key=lambda x: x["params_mb"])
    front[0]["crowding_distance"] = float("inf")
    front[-1]["crowding_distance"] = float("inf")
    p_min, p_max = front[0]["params_mb"], front[-1]["params_mb"]
    p_range = p_max - p_min if p_max > p_min else 1e-9
    for i in range(1, len(front) - 1):
        front[i]["crowding_distance"] += (
            (front[i + 1]["params_mb"] - front[i - 1]["params_mb"]) / p_range
        )


# ============ 8) 逐层 Evolutionary Search（前缀截断评估版） ============

class LayerwiseBlockES:
    """
    逐层贪心：在第 b 层开小EA，仅搜索该层算子（若为该stage头部可同时搜宽度）。
    baseline/candidate 都用"前缀截断模型"评估 SWAP（不再用 skip_connect 占位）。
    支持 pareto front 多目标优化（SWAP ↑, ParamsMB ↓）。
    """
    def __init__(self, search_space: MobileNetSearchSpace, swap_metric: SWAP, device,
                 num_inits=1, population_size=16, mutation_rate=0.3, n_generations=8,
                 stagewise_width_search=True, use_pareto=True):
        self.sp = search_space
        self.swap = swap_metric
        self.device = device
        self.num_inits = num_inits
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.stagewise_width_search = stagewise_width_search
        self.use_pareto = use_pareto  # 是否使用 pareto front

        # block -> stage 映射
        self.block2stage = []
        for s_idx, (_, _, n, _) in enumerate(self.sp.stage_setting):
            self.block2stage += [s_idx] * n
        assert len(self.block2stage) == self.sp.total_blocks

        self.skip_idx = self.sp._default_op_list().index("skip_connect")  # 仅用于最终补齐
        self.default_width_idx = self.sp.width_choices.index(1.0) if 1.0 in self.sp.width_choices else 0

    def _is_stage_head(self, b):
        s = self.block2stage[b]
        first = sum(self.sp.stage_setting[k][2] for k in range(s))
        return b == first

    def _score_prefix_pair(self, fixed_ops, fixed_wds, b_idx, op_gene, wd_gene, inputs):
        """
        返回 (fitness_log_delta, abs_swap_cand, params_mb_cand)
        基线 = 前缀到 b 层（不含 b），候选 = 前缀到 b 层 + 本候选 block
        """
        # 宽度代码更新（仅在 stage 头部时生效）
        width_codes = fixed_wds[:]
        if wd_gene is not None:
            stage_idx = self.block2stage[b_idx]
            width_codes[stage_idx] = wd_gene

        # 1) baseline：前缀到 b 层
        base_model = self.sp.get_prefix_model(fixed_ops, width_codes).to(self.device)
        for p in base_model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        if is_degenerate_head(base_model, inputs, self.device):
            swap_base = 0.0
        else:
            swap_base = float(np.mean([
                self.swap.evaluate(base_model, inputs) for _ in range(self.num_inits)
            ]))

        # 2) candidate：前缀到 b 层 + 候选 block
        cand_ops = fixed_ops + [op_gene]
        cand_model = self.sp.get_prefix_model(cand_ops, width_codes).to(self.device)
        for p in cand_model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        if is_degenerate_head(cand_model, inputs, self.device):
            swap_cand = 0.0
        else:
            swap_cand = float(np.mean([
                self.swap.evaluate(cand_model, inputs) for _ in range(self.num_inits)
            ]))

        # 使用原始（绝对）SWAP作为搜索fitness
        fitness_abs = swap_cand

        params_mb = count_parameters_in_MB(cand_model)  # 前缀模型参数量
        return fitness_abs, swap_cand, float(params_mb)

    def _per_block_ea(self, fixed_ops, fixed_wds, b_idx, inputs, baseline_swap):
        stage_idx = self.block2stage[b_idx]
        op_space = list(range(len(self.sp.op_list)))

        wd_space = None
        if self.stagewise_width_search and self._is_stage_head(b_idx):
            wd_space = list(range(len(self.sp.width_choices)))

        # 初始化
        pop = []
        for _ in range(self.population_size):
            op_gene = random.choice(op_space)
            wd_gene = random.choice(wd_space) if wd_space is not None else None
            fitness_abs, abs_swap, params_mb = self._score_prefix_pair(fixed_ops, fixed_wds, b_idx, op_gene, wd_gene, inputs)
            pop.append({
                "op_gene": op_gene, 
                "wd_gene": wd_gene, 
                "fitness": fitness_abs,
                "absolute_swap": abs_swap, 
                "params_mb": params_mb,
                "rank": 0,  # 初始化 rank
                "crowding_distance": 0.0  # 初始化 crowding_distance
            })

        # 迭代
        for g in range(self.n_generations):
            if self.use_pareto:
                # 使用 pareto front 排序
                fronts = non_dominated_sort_dict(pop)
                for f in fronts:
                    crowding_distance_dict(f)
                
                # 选择精英（从 pareto front 0 开始，按拥挤距离排序）
                elites = []
                for f in fronts:
                    if len(elites) + len(f) <= self.population_size // 2:
                        elites.extend(f)
                    else:
                        f_sorted = sorted(f, key=lambda x: x.get("crowding_distance", 0.0), reverse=True)
                        remaining = self.population_size // 2 - len(elites)
                        elites.extend(f_sorted[:remaining])
                        break
                
                # 记录 pareto front 0 的信息
                if len(fronts) > 0 and len(fronts[0]) > 0:
                    front0_sorted = sorted(fronts[0], key=lambda x: (-x["absolute_swap"], x["params_mb"]))
                    best_in_front0 = front0_sorted[0]
                    logging.info(
                        f"  [Block-{b_idx+1}] Generation {g+1}/{self.n_generations} | "
                        f"Front0_size={len(fronts[0])} | "
                        f"best_swap={best_in_front0['absolute_swap']:.1f}, "
                        f"best_params={best_in_front0['params_mb']:.3f}MB"
                    )
            else:
                # 单目标优化（仅按 SWAP）
                pop.sort(key=lambda ind: ind["fitness"], reverse=True)
                elites = pop[: self.population_size // 2]
                logging.info(f"  [Block-{b_idx+1}] Generation {g+1}/{self.n_generations} | best_abs={pop[0]['absolute_swap']:.1f}")
            
            # 生成下一代
            next_gen = elites[:]
            while len(next_gen) < self.population_size:
                p = random.choice(elites)
                child = dict(p)
                if random.random() < self.mutation_rate:
                    child["op_gene"] = self._rand_mutate(child["op_gene"], len(op_space))
                if wd_space is not None and random.random() < self.mutation_rate:
                    child["wd_gene"] = self._rand_mutate(child["wd_gene"], len(wd_space))
                fitness_abs, abs_swap, params_mb = self._score_prefix_pair(fixed_ops, fixed_wds, b_idx, child["op_gene"], child["wd_gene"], inputs)
                child["fitness"] = fitness_abs
                child["absolute_swap"] = abs_swap
                child["params_mb"] = params_mb
                child["rank"] = 0  # 重置 rank（会在下次 pareto 排序时更新）
                child["crowding_distance"] = 0.0  # 重置 crowding_distance
                next_gen.append(child)
            pop = next_gen

        # 选择最终最优解
        if self.use_pareto:
            # 使用 pareto front 选择：优先选择 SWAP 最大且 params 较小的
            fronts = non_dominated_sort_dict(pop)
            if len(fronts) > 0 and len(fronts[0]) > 0:
                front0_sorted = sorted(fronts[0], key=lambda x: (-x["absolute_swap"], x["params_mb"]))
                return front0_sorted[0]
            else:
                # 如果 pareto front 为空，回退到单目标
                pop.sort(key=lambda ind: ind["fitness"], reverse=True)
                return pop[0]
        else:
            pop.sort(key=lambda ind: ind["fitness"], reverse=True)
            return pop[0]

    @staticmethod
    def _rand_mutate(idx, space):
        x = random.randrange(space)
        if x == idx: x = (x + 1) % space
        return x

    def search(self, inputs, n_blocks_to_search=None):
        if n_blocks_to_search is None:
            n_blocks_to_search = self.sp.total_blocks

        fixed_ops = []
        fixed_wds = [self.default_width_idx] * len(self.sp.stage_setting)
        history = []

        # 计算初始 baseline（前缀长度=0 的模型）
        base0 = self.sp.get_prefix_model([], fixed_wds).to(self.device)
        for p in base0.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        baseline_swap = float(np.mean([self.swap.evaluate(base0, inputs)]))
        logging.info(f"Initial baseline SWAP (prefix=0 blocks): {baseline_swap:.1f}")

        for b in range(n_blocks_to_search):
            logging.info(f"=== Layer-wise EA on Block {b+1}/{n_blocks_to_search} ===")
            logging.info(f"  Current baseline SWAP (prefix={b}): {baseline_swap:.1f}")

            best = self._per_block_ea(fixed_ops, fixed_wds, b, inputs, baseline_swap)

            # 固定本层最优
            fixed_ops.append(best["op_gene"])
            if self._is_stage_head(b) and best["wd_gene"] is not None:
                fixed_wds[self.block2stage[b]] = best["wd_gene"]

            # 以“前缀到 b+1 层”的模型刷新 baseline
            prefix_model = self.sp.get_prefix_model(fixed_ops, fixed_wds).to(self.device)
            for p in prefix_model.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p)
            fit = float(np.mean([self.swap.evaluate(prefix_model, inputs)]))
            params_mb = count_parameters_in_MB(prefix_model)
            swap_delta_abs = fit - baseline_swap

            logging.info(
                f"[Layer-ES] Block {b+1} fixed: op={best['op_gene']}, "
                f"width_stage={self.block2stage[b]}->{fixed_wds[self.block2stage[b]] if self._is_stage_head(b) else 'NA'}, "
                f"SWAP(prefix)={fit:.1f}, delta_abs={swap_delta_abs:.1f}, "
                f"Params(prefix)={params_mb:.2f}MB"
            )

            history.append({
                "block": b,
                "op": best["op_gene"],
                "width_codes": fixed_wds[:],
                "fitness_prefix": fit,
                "swap_delta_abs": swap_delta_abs,
                "params_mb_prefix": params_mb
            })

            baseline_swap = fit  # 更新 baseline

        # 产出最终“前缀架构”（不补齐剩余 blocks）并计算一次 SWAP 与 Params，仅做记录
        final_ops_prefix = fixed_ops[:]
        final_wds = fixed_wds[:]

        prefix_model_final = self.sp.get_prefix_model(final_ops_prefix, final_wds).to(self.device)
        for p in prefix_model_final.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        final_fit_prefix = float(np.mean([self.swap.evaluate(prefix_model_final, inputs)]))
        final_params_prefix = count_parameters_in_MB(prefix_model_final)

        return {
            "op_codes_prefix": final_ops_prefix,
            "width_codes": final_wds,
            "fitness": final_fit_prefix,
            "params_mb_prefix": final_params_prefix,
            "history": history
        }


# ============ 9) 最终训练（Cutout/Mixup/LS/Cosine） ============

def train_and_eval(model, train_loader, val_loader, test_loader, device, args, rank=0):
    epochs = args.train_epochs
    lr = args.lr
    mixup_alpha = args.mixup_alpha
    label_smoothing = args.label_smoothing
    weight_decay = args.weight_decay

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0
        correct_top1, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if mixup_alpha > 0.:
                mixed_x, y_a, y_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
                outputs = model(mixed_x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct_top1 += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc_top1 = correct_top1 / total if total > 0 else 0.
        
        # Validation only on rank 0
        val_top1, val_top5 = 0.0, 0.0
        if rank == 0:
            val_top1, val_top5 = evaluate(model, val_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if rank == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                         f"Loss={train_loss:.3f}, "
                         f"LR={current_lr:.5f}, "
                         f"Train@1={train_acc_top1*100:.2f}%, "
                         f"Val@1={val_top1*100:.2f}%, Val@5={val_top5*100:.2f}%")

    final_top1 = 0.0
    if rank == 0:
        final_top1, final_top5 = evaluate(model, test_loader, device)
        logging.info(f"Final Test Accuracy: Top1={final_top1*100:.2f}%, Top5={final_top5*100:.2f}%")
        
    return final_top1


# ============ 10) CIFAR-10 DataLoader（最终训练用） ============

def get_cifar10_dataloaders(root, batch_size, num_workers=2,
                             use_cutout=False, cutout_length=16,
                             val_ratio=0.1, distributed=False):
    transform_list = [
        transforms.RandAugment(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if use_cutout:
        transform_list.append(Cutout(n_holes=1, length=cutout_length))
    transform_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465],
                                               [0.2023, 0.1994, 0.2010]))
    transform_train = transforms.Compose(transform_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
    ])

    full_train_ds_aug = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    full_train_ds_plain = datasets.CIFAR10(root, train=True, download=True, transform=transform_test)

    total_count = len(full_train_ds_aug)  # 50k
    val_count = int(total_count * val_ratio)
    generator = torch.Generator(); generator.manual_seed(0)
    indices = torch.randperm(total_count, generator=generator).tolist()
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_ds = torch.utils.data.Subset(full_train_ds_aug, train_indices)
    val_ds   = torch.utils.data.Subset(full_train_ds_plain, val_indices)
    test_ds  = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)

    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader




def setup_logger(log_path, log_name):
    os.makedirs(log_path, exist_ok=True)
    if logging.getLogger().handlers:
        for h in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s INFO: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_path, log_name), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )


