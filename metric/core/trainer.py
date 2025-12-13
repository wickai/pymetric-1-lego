#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, parameter_count_table
import os

import numpy as np
import metric.core.builders as builders
import metric.core.checkpoint as checkpoint
import metric.core.config as config
import metric.core.distributed as dist
import metric.core.logging as logging
import metric.core.meters as meters
import metric.core.net as net
import metric.core.optimizer as optim
import metric.core.ema as ema
from metric.core.ema import ema_avg_fn
import metric.datasets.loader as loader
import torch
from metric.core.config import cfg
from torch.optim.swa_utils import AveragedModel
import copy



logger = logging.get_logger(__name__)
# logger = setup_logger("./logs")


def get_model_complexity_info(model, inputs):
    """
    返回模型的参数数量（params）、激活数（acts）、FLOPs（flops），单位为原始数值（非MB、非GFLOPs）。
    Args:
        model: PyTorch 模型
        inputs: 输入样本（如 torch.randn(1, 3, 224, 224)）
    Returns:
        dict, 包含 "params", "acts", "flops"
    """
    # 计算 FLOPs 和激活数

    flops = FlopCountAnalysis(model, inputs)
    op_flops = flops.by_operator()
    total_conv_fc_flops = op_flops["conv"] + op_flops["linear"]

    acts = ActivationCountAnalysis(model, inputs)

    # 计算可训练参数数目
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "params": num_params,
        "acts": acts.total(),
        "flops": flops.total(),
        "total_conv_fc_flops": total_conv_fc_flops,
        "op_flops":op_flops
    }


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model(ema_flag=False):
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_arch()
    logger.info("Model:\n{}".format(model))
    # Log model complexity
    # logger.info(logging.dump_log_data(net.complexity(model), "complexity"))

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            dummy_target = torch.zeros(
                x.shape[0], dtype=torch.long, device=x.device)  # 根据你的任务适配
            return self.model(x, dummy_target)
    wrapped_model = Wrapper(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    stats = get_model_complexity_info(wrapped_model, dummy_input)
    logger.info(logging.dump_log_data(stats, "complexity"))

    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    ema_model = None
    if ema_flag:
        # ema model defination
        ema_model = AveragedModel(model, avg_fn=ema_avg_fn)
        ema_model = ema_model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=True,
        )
        # Set complexity function to be module's complexity function
        # model.complexity = model.module.complexity
        if ema_flag:
            ema_model = ema.EmaDDPWrapper(ema_model, device_ids=[cur_device], output_device=cur_device)
   
    return model, ema_model


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, ema_model=None):
    # return
    """Performs one epoch of training."""
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    if ema_model:
        ema_model.train()
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        # assert cfg.MODEL.TYPE.endswith(
        #     "_supermodel"
        # ), "train_code for retrain supuremodel only"
        if cfg.MODEL.TYPE.endswith("_supermodel"):
            if cfg.MODEL.SUPERMODELRETRAIN is True:
                # retrain mock model with rngs
                rng = cfg.MODEL.RANS  # use rngs from search methods.
            else:
                # train supermodel
                # Uniform Sampling
                rng = []
                operations = [list(range(5)) for i in range(21)]  # fixed
                for i, ops in enumerate(operations):
                    k = np.random.randint(len(ops))
                    select_op = ops[k]
                    rng.append(select_op)
                # logits = model(image, rng)
                rnd_labels = np.random.randint(0, 1000, len(labels))
                labels = rnd_labels
            logits, preds, targets = model(inputs, labels, rng)
        else:
            logits, preds, targets = model(inputs, labels)
        # Compute the loss
        loss = loss_fun(logits, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)
        # Compute the errors

        # add mix to modify these
        if cfg.DATA_LOADER.MIXUP_ALPHA == 0:
            top1_err, top5_err = meters.topk_errors(logits, labels, [1, 5])
            # Combine the stats across the GPUs (no reduction if 1 GPU used)
            loss, top1_err, top5_err = dist.scaled_all_reduce(
                [loss, top1_err, top5_err]
            )
            # Copy the stats from GPU to CPU (sync point)
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            train_meter.iter_toc()
            # Update and log stats
            mb_size = inputs.size(0) * cfg.NUM_GPUS
            train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        else:
            loss = dist.scaled_all_reduce(
                [
                    loss,
                ]
            )[0]
            loss = loss.item()
            train_meter.iter_toc()
            # Update and log stats
            mb_size = inputs.size(0) * cfg.NUM_GPUS
            train_meter.update_stats(loss, lr, mb_size)

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    # rng = []
    # operations = [list(range(5)) for i in range(21)]  # fixed
    # for i, ops in enumerate(operations):
    #     k = np.random.randint(len(ops))
    #     # k = 0
    #     select_op = ops[k]
    #     rng.append(select_op)
    # print(f"eval on random fix rngs:{rng}")

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        # assert cfg.MODEL.TYPE.endswith(
        #     "_supermodel"
        # ), "test_code for retrain supuremodel only"
        if cfg.MODEL.TYPE.endswith("_supermodel"):
            if cfg.MODEL.SUPERMODELRETRAIN is True:
                rng = cfg.MODEL.RANS  # use rngs from search methods.
            else:
                # Uniform Sampling
                rng = []
                operations = [list(range(5)) for i in range(21)]  # fixed
                for i, ops in enumerate(operations):
                    k = np.random.randint(len(ops))
                    k = 0  # use the left-most branch to eval
                    select_op = ops[k]
                    rng.append(select_op)

            # logits = model(image, rng)
            logits, preds, targets = model(inputs, labels, rng)
        else:
            logits, preds, targets = model(inputs, labels)
        # logits, preds, targets = model(inputs, labels)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(logits, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    
    model, ema_model = setup_model(ema_flag=cfg.TRAIN.EMA_FLAG)
    
    
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(
            last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(
            cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    # Compute model and loader timings
    # if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
    # benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, loss_fun,
                    optimizer, train_meter, cur_epoch, ema_model=ema_model)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(
                model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, test_meter, cur_epoch)
            # 评估 EMA 模型（仅主进程）
            if ema_model is not None:
                logger.info("Evaluating EMA model...")
                test_epoch(test_loader, ema_model, test_meter, cur_epoch)


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model, ema_model = setup_model(ema_flag=cfg.TRAIN.EMA_FLAG)
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    setup_model()
    builders.build_loss_fun().cuda()
    # Create data loaders
    loader.construct_train_loader()
    loader.construct_test_loader()
    # Compute model and loader timings
    # benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
