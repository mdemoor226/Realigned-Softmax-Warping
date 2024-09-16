#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.221.74.128" --master_port=12345 train.py configs/ms1mv3_r100_lr02
