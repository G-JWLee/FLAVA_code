#!/usr/bin/env python
import time

import torch
import torch.nn.parallel
import torch.distributed as dist
from numpy import inf

from utils import meters
from models.model_utils import epoch_wrapup


def calculate_throughput(loader, model, optimizer, loss_scaler, args, writer):

    # Make sure distributed sampler uses different samples in each process.
    loader.batch_sampler.set_epoch(epoch=0)
    optimizer.zero_grad()

    if args.optim.use_grad_clip:
        clip_grad = args.optim.clip_grad
    else:
        clip_grad = None

    # Accumulate gradient since the model is too big
    batch_size = args.optim.batch_size
    per_gpu_batchsize = args.optim.per_gpu_batchsize
    grad_steps = max(batch_size // per_gpu_batchsize, 1)
    accumulated_steps = 0

    # Switch to train mode
    model.train()

    optimizer.zero_grad()
    total_duration = 0

    # Log penalty loss
    if hasattr(model, 'module'):
        if model.module._req_opt:
            model.module.optimizer = optimizer
    else:
        if model._req_opt:
            model.optimizer = optimizer

    for data_idx, data in enumerate(loader):
        batch_i = loader.batch_sampler.advance_batches_seen() - 1
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        keys = set([k for k in data.keys() if "video" in k or "audio" in k])
        data = {k: v.cuda(args.environment.gpu, non_blocking=True) for k,v in data.items() if k in keys}

        start = time.time()
        # Compute output and loss
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = sum([v for k, v in output.items() if "loss" in k])

        # normalize loss to account for batch accumulation
        loss = loss / grad_steps
        # weight update
        accumulated_steps += 1
        # Update weight when gradient is accumulated or when the last sample is out
        update_grad = accumulated_steps % grad_steps == 0 or batch_i == len(loader) - 1
        loss_scaler(loss, optimizer, clip_grad=clip_grad, model=model, update_grad=update_grad)

        torch.cuda.synchronize()
        if update_grad:
            optimizer.zero_grad()
        duration = time.time() - start
        total_duration += duration

    throughput = args.data.args.num_videos / total_duration

    writer.add_text('throughput',str(throughput))
    writer.flush()

    print("Through put:", throughput,"sample/sec")




