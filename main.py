import os
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from models import GroupViT, MyModel, get_classifier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import yaml_config_hook, train
from sklearn.model_selection import KFold
from datasets import AbideFrameDataset, Transform


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_transforms = Transform(args.image_size, training=True)
    test_transforms = Transform(args.image_size, training=False)

    # load data file
    csv_file = pd.read_csv(args.csv_path)
    kf = KFold(n_splits=args.KFold, shuffle=True, random_state=args.seed)
    unique_patient = pd.unique(csv_file['SUB_ID'])
    # split the dataset based on patients
    for i, (train_id, test_id) in enumerate(kf.split(unique_patient)):
        # run only on one fold
        if args.fold is not None and i != args.fold:
            continue
        train_patient_idx = unique_patient[train_id]
        test_patient_idx = unique_patient[test_id]
        train_csv = csv_file[csv_file['SUB_ID'].isin(train_patient_idx)]
        test_csv = csv_file[csv_file['SUB_ID'].isin(test_patient_idx)]

        train_dataset = AbideFrameDataset(train_csv, args.data_root, task=args.task, transforms=train_transforms)
        step_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)

        # set sampler for parallel training
        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
            )
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        if rank == 0:
            # FIXME: only for faster evaluation, remember to disable the shuffle for full evaluation
            test_dataset = AbideFrameDataset(test_csv, args.data_root, task=args.task, transforms=train_transforms)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        else:
            test_loader = None

        loaders = (train_loader, test_loader)
        n_classes = train_dataset.n_classes

        encoder = GroupViT(img_size=args.image_size, patch_size=args.patch_size)
        n_features = encoder.num_features
        classifier = get_classifier(n_features, n_classes)
        model = MyModel(encoder, classifier).cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.scheduler:
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs * step_per_epoch, args.epochs * step_per_epoch)
        else:
            scheduler = None

        if args.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

        train(loaders, model, optimizer, scheduler, args, wandb_logger)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./configs/ABIDEI_DX.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb
    if not args.debug:
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

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