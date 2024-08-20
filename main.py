import os
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from models import get_classifier, get_encoder, get_aggregator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import yaml_config_hook, train
from sklearn.model_selection import KFold
from datasets import AbideFrameDataset, FrameTransform, FmriTransform, AbideFmriDataset


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

    frame_transforms = FrameTransform(args.image_size, training=True)
    fmri_train_transforms = FmriTransform(args.image_size, training=True)
    fmri_test_transforms = FmriTransform(args.image_size, training=False)

    # load data file
    frame_csv_file = pd.read_csv(args.frame_csv_path)
    fmri_csv_file = pd.read_csv(args.fmri_csv_path)

    kf = KFold(n_splits=args.KFold, shuffle=True, random_state=args.seed)
    unique_patient = pd.unique(frame_csv_file['SUB_ID'])

    # split the dataset based on patients
    for i, (train_id, test_id) in enumerate(kf.split(unique_patient)):
        # run only on one fold
        if args.fold is not None and i != args.fold:
            continue
        train_patient_idx = unique_patient[train_id]
        test_patient_idx = unique_patient[test_id]

        train_frame_csv = csv_file[frame_csv_file['SUB_ID'].isin(train_patient_idx)]
        train_frame_dataset = AbideFrameDataset(train_frame_csv, args.data_root, task=args.task, transforms=frame_transforms)

        train_fmri_csv = fmri_csv_file[fmri_csv_file['SUB_ID'].isin(train_patient_idx)]
        test_fmri_csv = fmri_csv_file[fmri_csv_file['SUB_ID'].isin(test_patient_idx)]s

        train_fmri_dataset = AbideFmriDataset(train_fmri_csv, args.data_root, task=args.task, transforms=fmri_train_transforms)

        # set sampler for parallel training
        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
            )
        else:
            train_sampler = None

        train_frame_loader = DataLoader(
            train_frame_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        train_fmri_loader = DataLoader(
            train_fmri_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )

        if rank == 0:
            test_fmri_dataset = AbideFmriDataset(test_fmri_csv, args.data_root, task=args.task, transforms=fmri_test_transforms)
            test_fmri_loader = DataLoader(test_fmri_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        else:
            test_fmri_loader = None

        n_classes = train_dataset.n_classes

        encoder = get_encoder(args)
        n_features = encoder.num_features
        aggregator = get_aggregator(args)
        classifier = get_classifier(n_features, n_classes)

        e_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr_e, weight_decay=args.weight_decay)
        m_optimizer = torch.optim.AdamW([{'params': aggregator.parameters(), 'params': classifier.parameters()}], lr=args.lr_m, weight_decay=args.weight_decay)

        if args.world_size > 1:
            encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
            aggregator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(aggregator)
            classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)

            encoder = DDP(encoder, device_ids=[gpu])
            aggregator = DDP(aggregator, device_ids=[gpu])
            classifier = DDP(classifier, device_ids=[gpu])

        dataloaders = (train_frame_loader, train_fmri_loader, test_fmri_loader)
        models = (encoder, aggregator, classifier)
        optimizers = (e_optimizer, m_optimizer)
        


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