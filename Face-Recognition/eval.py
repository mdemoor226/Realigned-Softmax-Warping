import argparse

import cv2
import numpy as np
import torch
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
import os
import logging

from backbones import get_model
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_config import get_config

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    print(feat)


if __name__ == "__main__":
    config_path = 'configs/ms1mv3_r100_lr02'
    cfg = get_config(config_path)
    init_logging(rank, cfg.output)
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    )
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )

    net = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).cuda()
    model_path = './work_dirs/ms1mv3_r100_lr02/model.pt'
    net.load_state_dict(torch.load(model_path))
    
    callback_verification(1, net)
    distributed.destroy_process_group()

