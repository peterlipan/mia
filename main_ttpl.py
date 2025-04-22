import os
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import yaml_config_hook, Trainer


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size, timeout=timedelta(hours=12))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = Trainer(args, wandb_logger)
    trainer.run_ttpl(args)



if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./configs/ABIDE.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()


    args.world_size = args.gpus * args.nodes

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # set number of rois according to the atlas
    atlas2roi = {'cc400': 392, 'ho': 111, 'cc200': 200, 'aal': 116} if 'ABIDE' in args.dataset else {'cc400': 351, 'cc200': 190, 'ho': 111, 'aal': 116}
    args.num_roi = atlas2roi[args.atlas]

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb
    if not args.debug:
        wandb.require("core")
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = vars(args)

        wandb_logger = wandb.init(
            project=f"{args.dataset}_{args.task}",
            config=config
        )
    else:
        wandb_logger = None

    if args.world_size > 1:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger,), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger)